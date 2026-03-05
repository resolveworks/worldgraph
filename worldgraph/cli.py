from pathlib import Path

import click

from worldgraph.extract import run_extraction
from worldgraph.match import run_matching


@click.group()
def cli():
    """Worldgraph — cross-source structural matching for knowledge extraction."""


@cli.command()
@click.argument(
    "articles", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("-o", "--output-dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    help="Claude model to use for extraction.",
)
def extract(articles: tuple[Path, ...], output_dir: Path, model: str):
    """Stage 1: Extract entities and relations from articles."""
    run_extraction(list(articles), output_dir, model)


@cli.command()
@click.argument(
    "graphs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
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
def match(
    graphs: tuple[Path, ...],
    output: Path,
    relation_threshold: float,
    match_threshold: float,
    max_iter: int,
):
    """Stage 2: Entity alignment via similarity propagation — merge matched graphs."""
    run_matching(
        list(graphs),
        output,
        relation_threshold=relation_threshold,
        match_threshold=match_threshold,
        max_iter=max_iter,
    )
