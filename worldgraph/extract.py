import os
import uuid
from pathlib import Path

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from worldgraph.graph import Graph, save_graph

load_dotenv()

SYSTEM_PROMPT = """You are an entity-relation extraction system building a graph of world facts from a news article. Entities are things in the world — people, organizations, places, and things — and relations are facts the article asserts between two distinct entities or relations.

Be thorough: capture every entity and every asserted fact. Use the exact names as they appear in the text.

Rules:
- Extract only what the article asserts as fact. Denied, disputed, or merely alleged claims are not extracted.
- Every relation connects two distinct entities or relations. If an action's object is a thing in the world, make it an entity and connect it ("used ecstasy" becomes a "used" relation to the entity "ecstasy"). An action with no entity object produces no relation.
- A relation's source and target may each be the id of an entity or of another relation. A qualifier of a fact, such as a role or place, is expressed as a relation whose source or target is the id of the relation it qualifies: if r1 states that Tessa Corin manages Halden Freight, her role is a relation with source r1, target a 'managing director' entity, and relation 'role'.
- The media is not part of the world graph: the publishing outlet, journalists, photographers, and the act of reporting never appear as entities or relations.
- A relation phrase contains only the relation itself, never entity names. Entities that a fact refers to are nodes, not phrase content.
- Write relation phrases in base form without tense: 'acquire', 'be headquartered in' — never 'acquired', 'will acquire', 'is headquartered in'. Tense marking is not captured.

Each entity should have a short unique id and the name as it appears in the text. Each relation should have a short unique id, e.g. 'r1', 'r2'."""


class Entity(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")


class Relation(BaseModel):
    id: str = Field(
        description="Short unique identifier for this relation, e.g. 'r1', 'r2'"
    )
    source: str = Field(
        description="The 'id' of the source entity or source relation"
    )
    target: str = Field(
        description="The 'id' of the target entity or target relation"
    )
    relation: str = Field(
        description="Base-form verb phrase without tense, e.g. 'acquire', 'be headquartered in' — "
        "never 'acquired', 'will acquire', or 'is headquartered in'"
    )


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]

    @model_validator(mode="after")
    def _validate_ids(self) -> "Extraction":
        """Entity and relation ids are globally unique and live in one shared
        reference namespace; every relation endpoint must resolve to an entity
        or a relation of this extraction. Forward references between
        relations are valid.
        """
        entity_ids = [entity.id for entity in self.entities]
        relation_ids = [relation.id for relation in self.relations]
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(f"duplicate entity ids: {sorted(entity_ids)}")
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError(f"duplicate relation ids: {sorted(relation_ids)}")
        shared = sorted(set(entity_ids) & set(relation_ids))
        if shared:
            raise ValueError(
                f"entity and relation ids must be disjoint, shared: {shared}"
            )
        known = set(entity_ids) | set(relation_ids)
        unknown = sorted(
            {
                ref
                for relation in self.relations
                for ref in (relation.source, relation.target)
                if ref not in known
            }
        )
        if unknown:
            raise ValueError(f"relations reference unknown ids: {unknown}")
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
    """Extract entities and relations from a single article's text."""
    prompt = f"""Extract all entities and relations from this news article.

{text}"""

    return agent.run_sync(prompt).output


def extraction_to_graph(article_id: str, extraction: Extraction) -> Graph:
    """Convert an extraction into a runtime graph, preserving every entity and
    relation id and every relation-to-relation reference.

    Runtime edge ids are pre-allocated so edge endpoints resolve regardless
    of relation order, including forward references.
    """
    graph = Graph(id=article_id)
    term_ids: dict[str, str] = {}
    for entity in extraction.entities:
        term_ids[entity.id] = graph.add_entity(entity.name).id

    edge_ids = {rel.id: str(uuid.uuid4()) for rel in extraction.relations}
    term_ids.update(edge_ids)
    for rel in extraction.relations:
        graph.add_edge(
            term_ids[rel.source],
            term_ids[rel.target],
            rel.relation,
            id=edge_ids[rel.id],
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
