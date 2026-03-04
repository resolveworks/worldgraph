import json
import logging
import uuid
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
        data = extraction.model_dump()

        # Replace LLM-generated entity IDs with UUIDs
        id_map = {e["id"]: str(uuid.uuid4()) for e in data["entities"]}

        nodes = []
        edges = []
        for entity in data["entities"]:
            new_id = id_map[entity["id"]]
            # Entity node (no "label" key)
            nodes.append({"id": new_id})
            # Literal node for the entity name
            label_id = str(uuid.uuid4())
            nodes.append({"id": label_id, "label": entity["name"]})
            # "is named" edge from entity to its literal
            edges.append(
                {
                    "source": new_id,
                    "target": label_id,
                    "relation": "is named",
                }
            )

        for rel in data["relations"]:
            if rel["source"] not in id_map or rel["target"] not in id_map:
                bad = [k for k in ("source", "target") if rel[k] not in id_map]
                logger.warning(
                    "Dropping relation %r — invalid %s: %s",
                    rel["relation"],
                    ", ".join(bad),
                    ", ".join(repr(rel[k]) for k in bad),
                )
                continue
            edges.append(
                {
                    "source": id_map[rel["source"]],
                    "target": id_map[rel["target"]],
                    "relation": rel["relation"],
                }
            )

        graph = {"id": article_id, "nodes": nodes, "edges": edges}
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)

    click.echo(f"\nWrote graphs to {graphs_dir}/")
