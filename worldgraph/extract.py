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


def run_extraction(input_path: Path, output_path: Path, model: str) -> None:
    """Run extraction on all articles and write graph JSON directly."""
    article_files = sorted(input_path.glob("*.json"))
    articles = []
    for f in article_files:
        with open(f) as fh:
            articles.append(json.load(fh))

    client = anthropic.Anthropic()
    graphs = []

    for i, article in enumerate(articles, 1):
        click.echo(f"[{i}/{len(articles)}] Extracting from: {article['title']}")
        extraction = extract_article(client, article, model)
        data = extraction.model_dump()

        article_id = article["id"]

        # Replace LLM-generated entity IDs with UUIDs
        id_map = {e["id"]: str(uuid.uuid4()) for e in data["entities"]}

        entities = []
        for entity in data["entities"]:
            new_id = id_map[entity["id"]]
            entities.append(
                {
                    "id": new_id,
                    "name": entity["name"],
                    "occurrences": [
                        {
                            "article_id": article_id,
                            "entity_id": new_id,
                            "name": entity["name"],
                        }
                    ],
                }
            )

        edges = []
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
                    "articles": [article_id],
                }
            )

        graphs.append({"id": article_id, "entities": entities, "edges": edges})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"graphs": graphs}, f, indent=2)

    click.echo(f"\nWrote {len(graphs)} article graphs to {output_path}")
