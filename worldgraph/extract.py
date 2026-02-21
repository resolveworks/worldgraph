import json
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

SYSTEM_PROMPT = """You are an entity-relation extraction system. Given a news article, extract all entities and the relations between them.

Be thorough: capture every entity and relation mentioned in the article. Use the exact names as they appear in the text. Each relation should be a concise verb phrase.

For entity types, use short descriptive phrases like "tech company", "CEO", "city", "acquisition price", "date", "AI startup", "energy company", etc. Be specific rather than generic."""


class Entity(BaseModel):
    id: str  # "e1", "e2", etc.
    name: str  # as it appears in the article
    type: str  # free-form: "tech company", "CEO", "city", etc.


class Relation(BaseModel):
    source: str  # entity ID
    relation: str  # free-form verb phrase
    target: str  # entity ID
    context: str  # supporting quote


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
        results.append(
            {
                "article_id": article["id"],
                "source": article["source"],
                "title": article["title"],
                "entities": data["entities"],
                "relations": data["relations"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nWrote {len(results)} extractions to {output_path}")
