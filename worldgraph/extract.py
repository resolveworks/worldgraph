import json
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv

load_dotenv()

EXTRACTION_TOOL = {
    "name": "extract_entities_and_relations",
    "description": "Extract all entities and their relations from a news article. Identify every person, organization, location, monetary amount, date, and other named entities mentioned. Then capture all relations between them.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "All entities mentioned in the article",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short unique identifier like e1, e2, etc.",
                        },
                        "name": {
                            "type": "string",
                            "description": "The entity name as it appears in the article",
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "person",
                                "organization",
                                "location",
                                "monetary_amount",
                                "date",
                                "product",
                                "event",
                                "other",
                            ],
                            "description": "The type of entity",
                        },
                    },
                    "required": ["id", "name", "type"],
                },
            },
            "relations": {
                "type": "array",
                "description": "All relations between entities",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Entity ID of the source/subject",
                        },
                        "relation": {
                            "type": "string",
                            "description": "The relation phrase (e.g. 'acquired', 'is CEO of', 'located in')",
                        },
                        "target": {
                            "type": "string",
                            "description": "Entity ID of the target/object",
                        },
                        "context": {
                            "type": "string",
                            "description": "Brief quote or paraphrase from the article supporting this relation",
                        },
                    },
                    "required": ["source", "relation", "target", "context"],
                },
            },
        },
        "required": ["entities", "relations"],
    },
}

SYSTEM_PROMPT = """You are an entity-relation extraction system. Given a news article, extract all entities (people, organizations, locations, amounts, dates, products, events) and the relations between them.

Be thorough: capture every entity and relation mentioned in the article. Use the exact names as they appear in the text. Each relation should be a concise verb phrase."""


def extract_article(client: anthropic.Anthropic, article: dict, model: str) -> dict:
    """Extract entities and relations from a single article."""
    prompt = f"""Extract all entities and relations from this news article.

Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['body']}"""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_entities_and_relations"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError(f"No tool_use block in response for article {article['id']}")


def run_extraction(input_path: Path, output_path: Path, model: str) -> None:
    """Run extraction on all articles and write results."""
    with open(input_path) as f:
        articles = json.load(f)

    client = anthropic.Anthropic()
    results = []

    for i, article in enumerate(articles, 1):
        click.echo(f"[{i}/{len(articles)}] Extracting from: {article['title']}")
        extraction = extract_article(client, article, model)
        results.append(
            {
                "article_id": article["id"],
                "source": article["source"],
                "title": article["title"],
                "entities": extraction["entities"],
                "relations": extraction["relations"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nWrote {len(results)} extractions to {output_path}")
