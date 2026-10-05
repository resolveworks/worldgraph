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
def extract(articles: tuple[Path, ...], output_dir: Path):
    """Stage 1: Extract term graphs from article text files (filename stem = article id)."""
    run_extraction(list(articles), output_dir)


@cli.command()
@click.argument(
    "graphs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
def match(graphs: tuple[Path, ...], output: Path):
    """Stage 2: Term alignment via similarity propagation — merge matched graphs."""
    run_matching(list(graphs), output)
