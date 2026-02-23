from pathlib import Path

import click

from worldgraph.extract import run_extraction
from worldgraph.match import run_matching


@click.group()
def cli():
    """Worldgraph — cross-source structural matching for knowledge extraction."""


@cli.command()
@click.option(
    "--input",
    "input_path",
    default="data/articles",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Input articles directory (one JSON file per article).",
)
@click.option(
    "--output",
    "output_path",
    default="data/graphs_0.json",
    type=click.Path(path_type=Path),
    help="Output graph JSON file.",
)
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    help="Claude model to use for extraction.",
)
def extract(input_path: Path, output_path: Path, model: str):
    """Stage 1: Extract entities and relations from articles."""
    run_extraction(input_path, output_path, model)


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    default="data/graphs_0.json",
    type=click.Path(exists=True, path_type=Path),
    help="Input graph JSON file.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default="data/graphs_1.json",
    type=click.Path(path_type=Path),
    help="Output graph JSON file.",
)
@click.option(
    "--threshold",
    default=0.8,
    type=float,
    help="Relative similarity threshold for entity matching (SelectThreshold).",
)
@click.option(
    "--max-iter",
    default=30,
    type=int,
    help="Maximum propagation iterations.",
)
def match(input_path: Path, output_path: Path, threshold: float, max_iter: int):
    """Stage 2: Entity alignment via similarity propagation — merge matched graphs."""
    run_matching(input_path, output_path, threshold, max_iter=max_iter)
