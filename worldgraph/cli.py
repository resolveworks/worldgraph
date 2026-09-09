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
    default="deepseek:deepseek-v4-flash",
    help="Model to use for extraction, as a provider-prefixed pydantic-ai string.",
)
def extract(articles: tuple[Path, ...], output_dir: Path, model: str):
    """Stage 1: Extract entities and relations from article text files (filename stem = article id)."""
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
    "--max-iter",
    default=30,
    type=int,
    help="Maximum propagation iterations.",
)
@click.option(
    "--merge-threshold",
    default=0.9,
    type=float,
    help="Minimum confidence, backed by structural evidence, to merge two entities.",
)
def match(
    graphs: tuple[Path, ...],
    output: Path,
    relation_threshold: float,
    max_iter: int,
    merge_threshold: float,
):
    """Stage 2: Entity alignment via similarity propagation — merge matched graphs."""
    run_matching(
        list(graphs),
        output,
        relation_threshold=relation_threshold,
        max_iter=max_iter,
        merge_threshold=merge_threshold,
    )
