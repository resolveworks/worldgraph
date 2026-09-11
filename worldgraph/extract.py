import os
import uuid
from pathlib import Path

import click
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from worldgraph.graph import Graph, save_graph

SYSTEM_PROMPT = """You extract world facts from a news article as a graph of terms. A term is an entity or a statement. Entities are named things in the world — people, organizations, places, things. Use the exact name as it appears in the text, without a leading article; unnamed mentions ('two intruders', 'a cleaner') never become entities. A statement is a triple: subject, predicate, object. Subject and object are ids of entities or of other statements. The predicate is a short verb phrase in active voice, and the subject is the one who brings the fact about: 'Acme acquired Beta', never the passive with swapped participants. Capture every fact the article asserts. Qualifiers of a fact — a title, a place, a price, a scope — are statements about that statement: a 'work at' fact with subject Jane and object Supercorp gets a second statement (that fact, 'as', CEO). Claims about claims are statements too, and you never judge their truth: a denial is (denier, 'denies', the denied statement); an allegation is (alleged-claimant, 'alleges', the claimed statement). Resolve pronouns to the entity they refer to. The media is not part of the world: the outlet, journalists, photographers, and the act of reporting never appear as terms. Entities get short unique ids like 'e1', 'e2'; statements like 's1', 's2'."""


class EntityRef(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")


class StatementModel(BaseModel):
    id: str = Field(
        description="Short unique identifier for this statement, e.g. 's1', 's2'"
    )
    subject: str = Field(
        description="The id of the entity or statement this statement is about "
        "— the one who brings the fact about"
    )
    predicate: str = Field(
        description="Short verb phrase in active voice, e.g. 'acquire', 'work at', 'deny'"
    )
    object: str = Field(
        description="The id of the entity or statement the fact is directed at"
    )


class Extraction(BaseModel):
    entities: list[EntityRef]
    statements: list[StatementModel]

    @model_validator(mode="after")
    def _validate_ids(self) -> "Extraction":
        """Entity and statement ids share one namespace and must be
        globally unique; every statement endpoint must resolve to an
        entity or statement of this extraction, and no statement may
        reference itself directly. Forward references between statements
        are valid.
        """
        entity_ids = [entity.id for entity in self.entities]
        statement_ids = [statement.id for statement in self.statements]
        all_ids = entity_ids + statement_ids
        duplicates = sorted({term_id for term_id in all_ids if all_ids.count(term_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate term ids: {duplicates}")
        known = set(all_ids)
        unknown = sorted(
            {
                endpoint
                for statement in self.statements
                for endpoint in (statement.subject, statement.object)
                if endpoint not in known
            }
        )
        if unknown:
            raise ValueError(f"statements reference unknown ids: {unknown}")
        self_refs = sorted(
            statement.id
            for statement in self.statements
            if statement.subject == statement.id or statement.object == statement.id
        )
        if self_refs:
            raise ValueError(f"statements referencing themselves: {self_refs}")
        return self


def build_agent(model: str) -> Agent[object, Extraction]:
    """Single source of truth for the extraction agent — used by the pipeline and the eval harness."""
    return Agent(
        model,
        instructions=SYSTEM_PROMPT,
        output_type=Extraction,
        model_settings={"thinking": "low"},
    )


def extract_article(agent: Agent[object, Extraction], text: str) -> Extraction:
    """Extract entities and statements from a single article's text."""
    prompt = f"""<article>
{text}
</article>

Extract the entities and statements the article asserts."""

    return agent.run_sync(prompt).output


def extraction_to_graph(article_id: str, extraction: Extraction) -> Graph:
    """Convert an extraction into a runtime graph: entities and statements
    become terms. Runtime ids are pre-allocated for every extraction term
    so statement endpoints resolve regardless of listing order."""
    graph = Graph(id=article_id)
    runtime_id = {
        term.id: str(uuid.uuid4())
        for term in (extraction.entities + extraction.statements)
    }
    for entity in extraction.entities:
        graph.add_entity(entity.name, id=runtime_id[entity.id])
    for statement in extraction.statements:
        graph.add_statement(
            runtime_id[statement.subject],
            statement.predicate,
            runtime_id[statement.object],
            id=runtime_id[statement.id],
        )

    graph.validate()
    return graph


def run_extraction(article_files: list[Path], output_dir: Path) -> None:
    """Run extraction on all article text files, writing one graph JSON per article.

    The filename stem is the article id.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(os.environ["EXTRACTION_MODEL"])

    for i, article_file in enumerate(article_files, 1):
        article_id = article_file.stem
        out_path = output_dir / f"{article_id}.json"
        if out_path.exists():
            click.echo(
                f"[{i}/{len(article_files)}] Skipping (already extracted): {article_file.name}"
            )
            continue

        click.echo(f"[{i}/{len(article_files)}] Extracting from: {article_file.name}")
        extraction = extract_article(agent, article_file.read_text())

        graph = extraction_to_graph(article_id, extraction)
        save_graph(graph, out_path)

    click.echo(f"\nWrote graphs to {output_dir}/")
