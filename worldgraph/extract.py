import logging
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from worldgraph.graph import Graph, save_graph

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an entity-relation extraction system building a graph of world facts from a news article. Entities are things in the world — people, organizations, places, and things — and relations are facts the article asserts between two distinct entities.

Be thorough: capture every entity and every asserted fact. Use the exact names as they appear in the text.

Rules:
- Extract only what the article asserts as fact. Denied, disputed, or merely alleged claims are not extracted.
- Every relation connects two distinct entities. If an action's object is a thing in the world, make it an entity and connect it ("used ecstasy" becomes a "used" relation to the entity "ecstasy"). An action with no entity object produces no relation.
- The media is not part of the world graph: the publishing outlet, journalists, photographers, and the act of reporting never appear as entities or relations.
- A relation phrase contains only the relation itself, never entity names. Entities that a fact refers to are nodes, not phrase content.
- Write relation phrases in base form without tense: 'acquire', 'be headquartered in' — never 'acquired', 'will acquire', 'is headquartered in'. Tense is carried by the temporal field, not the phrase.
- Every relation has a temporal dimension, read from the article's own wording: 'past' for facts presented as completed or no longer true, 'current' for facts stated as true now, 'future' for announced, planned, or expected facts. When the wording does not mark time, the fact is 'current'.

Each entity should have a short unique id and the name as it appears in the text."""


class Entity(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")


Temporal = Literal["past", "current", "future"]


class Relation(BaseModel):
    source: str = Field(description="The 'id' of the source entity")
    target: str = Field(description="The 'id' of the target entity")
    relation: str = Field(
        description="Base-form verb phrase without tense, e.g. 'acquire', 'be headquartered in' — "
        "never 'acquired', 'will acquire', or 'is headquartered in'. Tense is carried by the temporal field."
    )
    temporal: Temporal = Field(
        description="When the relation holds, read from the article's own wording: "
        "'past' — presented as completed or no longer true ('acquired', 'former', 'previously'); "
        "'current' — stated as true now, the default when the wording does not mark time; "
        "'future' — announced, planned, or expected ('will', 'plans to', 'is expected to')"
    )


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
            graph.add_edge(
                entity_map[rel.source], entity_map[rel.target], rel.relation, rel.temporal
            )

        save_graph(graph, out_path)

    click.echo(f"\nWrote graphs to {output_dir}/")
