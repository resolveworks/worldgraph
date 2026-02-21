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

For entity types, use short descriptive phrases like "tech company", "CEO", "city", "acquisition price", "date", "AI startup", "energy company", etc. Be specific rather than generic."""


class Entity(BaseModel):
    id: str = Field(description="Short unique identifier for this entity, e.g. 'e1', 'e2'")
    name: str = Field(description="Entity name as it appears in the article")
    type: str = Field(description="Short type phrase, e.g. 'tech company', 'CEO', 'city'")


class Relation(BaseModel):
    source: str = Field(description="The 'id' of the source entity")
    target: str = Field(description="The 'id' of the target entity")
    relation: str = Field(description="Concise verb phrase describing the relation")
    context: str = Field(description="Supporting quote from the article")


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


def extract_article(client: anthropic.Anthropic, article: dict, model: str) -> Extraction:
    """Extract entities and relations from a single article."""
    prompt = f"""Extract all entities and relations from this news article.

Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['body']}"""

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=Extraction,
    )

    return response.parsed_output


def run_extraction(input_path: Path, output_path: Path, model: str) -> None:
    """Run extraction on all articles and write results."""
    with open(input_path) as f:
        articles = json.load(f)

    client = anthropic.Anthropic()
    results = []

    for i, article in enumerate(articles, 1):
        click.echo(f"[{i}/{len(articles)}] Extracting from: {article['title']}")
        extraction = extract_article(client, article, model)
        data = extraction.model_dump()

        # Replace LLM-generated entity IDs with UUIDs
        id_map = {e["id"]: str(uuid.uuid4()) for e in data["entities"]}
        for entity in data["entities"]:
            entity["id"] = id_map[entity["id"]]
        remapped_relations = []
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
            rel["source"] = id_map[rel["source"]]
            rel["target"] = id_map[rel["target"]]
            remapped_relations.append(rel)

        results.append(
            {
                "article_id": article["id"],
                "source": article["source"],
                "title": article["title"],
                "entities": data["entities"],
                "relations": remapped_relations,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nWrote {len(results)} extractions to {output_path}")
