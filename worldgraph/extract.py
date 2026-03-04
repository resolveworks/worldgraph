import json
import logging
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from worldgraph.constants import EntityType
from worldgraph.graph import Graph, save_graph

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an entity-relation extraction system. Given a news article, extract all entities and the relations between them.

Be thorough: capture every entity and relation mentioned in the article. Use the exact names as they appear in the text. Each relation should be a concise verb phrase.

Each entity should have a short unique id, the name as it appears in the text, and a type classification. Valid types are: person, organization, location, event, concept."""


class Entity(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")
    type: EntityType = Field(
        description="Entity type: person, organization, location, event, or concept"
    )


class Relation(BaseModel):
    source: str = Field(description="The 'id' of the source entity")
    target: str = Field(description="The 'id' of the target entity")
    relation: str = Field(description="Concise verb phrase describing the relation")


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


def extract_article(
    client: anthropic.Anthropic, article: dict, model: str
) -> Extraction:
    """Extract entities and relations from a single article."""
    prompt = f"""Extract all entities and relations from this news article.

Title: {article["title"]}
Source: {article["source"]}
Date: {article["date"]}

{article["body"]}"""

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=Extraction,
    )

    return response.parsed_output


def run_extraction(articles_dir: Path, graphs_dir: Path, model: str) -> None:
    """Run extraction on all articles, writing one graph JSON per article."""
    article_files = sorted(articles_dir.glob("*.json"))
    articles = []
    for f in article_files:
        with open(f) as fh:
            articles.append(json.load(fh))

    graphs_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    for i, article in enumerate(articles, 1):
        article_id = article["id"]
        out_path = graphs_dir / f"{article_id}.json"
        if out_path.exists():
            click.echo(
                f"[{i}/{len(articles)}] Skipping (already extracted): {article['title']}"
            )
            continue

        click.echo(f"[{i}/{len(articles)}] Extracting from: {article['title']}")
        extraction = extract_article(client, article, model)

        # Build graph using shared data model
        graph = Graph(id=article_id)
        entity_map = {}
        for entity in extraction.entities:
            node = graph.add_entity(entity.name, entity.type)
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

    click.echo(f"\nWrote graphs to {graphs_dir}/")
