from pathlib import Path

import click

from worldgraph.extract import run_extraction
from worldgraph.match import run_matching

ARTICLES_DIR = Path("data/articles")
GRAPHS_DIR = Path("data/graphs")
MATCHED_PATH = Path("data/matched.json")


@click.group()
def cli():
    """Worldgraph — cross-source structural matching for knowledge extraction."""


@cli.command()
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    help="Claude model to use for extraction.",
)
def extract(model: str):
    """Stage 1: Extract entities and relations from articles into data/graphs/."""
    run_extraction(ARTICLES_DIR, GRAPHS_DIR, model)


@cli.command()
@click.option(
    "--relation-threshold",
    default=0.8,
    type=float,
    help="Minimum cosine similarity for two relation phrases to be treated as equivalent.",
)
@click.option(
    "--match-threshold",
    default=0.8,
    type=float,
    help="Minimum confidence score to merge two entities.",
)
@click.option(
    "--max-iter",
    default=30,
    type=int,
    help="Maximum propagation iterations.",
)
def match(relation_threshold: float, match_threshold: float, max_iter: int):
    """Stage 2: Entity alignment via similarity propagation — merge matched graphs."""
    run_matching(
        GRAPHS_DIR,
        MATCHED_PATH,
        relation_threshold=relation_threshold,
        match_threshold=match_threshold,
        max_iter=max_iter,
    )
