import logging
from pathlib import Path

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from worldgraph.graph import Graph, save_graph

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an entity-relation extraction system. Given a news article, extract all entities and the relations between them.

Be thorough: capture every entity and relation mentioned in the article. Use the exact names as they appear in the text. Each relation should be a concise verb phrase.

Each entity should have a short unique id and the name as it appears in the text."""


class Entity(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")


class Relation(BaseModel):
    source: str = Field(description="The 'id' of the source entity")
    target: str = Field(description="The 'id' of the target entity")
    relation: str = Field(description="Concise verb phrase describing the relation")


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


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


def run_extraction(article_files: list[Path], output_dir: Path, model: str) -> None:
    """Run extraction on all article text files, writing one graph JSON per article.

    The filename stem is the article id.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(model)

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

        # Build graph using shared data model
        graph = Graph(id=article_id)
        entity_map = {}
        for entity in extraction.entities:
            node = graph.add_entity(entity.name)
            entity_map[entity.id] = node

        for rel in extraction.relations:
            if rel.source not in entity_map or rel.target not in entity_map:
                bad = [
                    k for k in ("source", "target") if getattr(rel, k) not in entity_map
                ]
                logger.warning(
                    "Dropping relation %r — invalid %s: %s",
                    rel.relation,
                    ", ".join(bad),
                    ", ".join(repr(getattr(rel, k)) for k in bad),
                )
                continue
            graph.add_edge(entity_map[rel.source], entity_map[rel.target], rel.relation)

        save_graph(graph, out_path)

    click.echo(f"\nWrote graphs to {output_dir}/")
