from pathlib import Path

import click

from worldgraph.constants import MERGE_THRESHOLD
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
def extract(articles: tuple[Path, ...], output_dir: Path):
    """Stage 1: Extract entities and events from article text files (filename stem = article id)."""
    run_extraction(list(articles), output_dir)


@cli.command()
@click.argument(
    "graphs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--max-iter",
    default=30,
    type=int,
    help="Maximum propagation iterations.",
)
@click.option(
    "--merge-threshold",
    default=MERGE_THRESHOLD,
    type=float,
    help="Minimum confidence, backed by structural evidence, to merge two nodes.",
)
def match(
    graphs: tuple[Path, ...],
    output: Path,
    max_iter: int,
    merge_threshold: float,
):
    """Stage 2: Term alignment via similarity propagation — merge matched graphs."""
    run_matching(
        list(graphs),
        output,
        max_iter=max_iter,
        merge_threshold=merge_threshold,
    )
