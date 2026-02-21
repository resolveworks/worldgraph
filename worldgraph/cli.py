from pathlib import Path

import click

from worldgraph.cluster import run_clustering
from worldgraph.extract import run_extraction
from worldgraph.match import run_matching


@click.group()
def cli():
    """Worldgraph — cross-source structural matching for knowledge extraction."""


@cli.command()
@click.option(
    "--input",
    "input_path",
    default="data/articles.json",
    type=click.Path(exists=True, path_type=Path),
    help="Input articles JSON file.",
)
@click.option(
    "--output",
    "output_path",
    default="data/extractions.json",
    type=click.Path(path_type=Path),
    help="Output extractions JSON file.",
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
    "--input",
    "input_path",
    default="data/extractions.json",
    type=click.Path(exists=True, path_type=Path),
    help="Input extractions JSON file.",
)
@click.option(
    "--output",
    "output_path",
    default="data/clusters.json",
    type=click.Path(path_type=Path),
    help="Output clusters JSON file.",
)
@click.option(
    "--model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Sentence embedding model name.",
)
@click.option(
    "--threshold",
    default=0.7,
    type=float,
    help="Cosine similarity threshold for clustering.",
)
def cluster(input_path: Path, output_path: Path, model: str, threshold: float):
    """Stage 2: Embed and cluster similar relations."""
    run_clustering(input_path, output_path, model, threshold)


@cli.command()
@click.option(
    "--extractions",
    "extractions_path",
    default="data/extractions.json",
    type=click.Path(exists=True, path_type=Path),
    help="Input extractions JSON file.",
)
@click.option(
    "--clusters",
    "clusters_path",
    default="data/clusters.json",
    type=click.Path(exists=True, path_type=Path),
    help="Input clusters JSON file.",
)
@click.option(
    "--output",
    "output_path",
    default="data/matches.json",
    type=click.Path(path_type=Path),
    help="Output matches JSON file.",
)
@click.option(
    "--name-threshold",
    default=0.5,
    type=float,
    help="Cosine similarity threshold for entity name matching.",
)
def match(
    extractions_path: Path,
    clusters_path: Path,
    output_path: Path,
    name_threshold: float,
):
    """Stage 3: Structural matching across article subgraphs."""
    run_matching(extractions_path, clusters_path, output_path, name_threshold)


@cli.command()
def merge():
    """Stage 4: Iterative entity merging. (not yet implemented)"""
    click.echo("Not yet implemented.")


@cli.command()
def score():
    """Stage 5: Score facts by cross-source agreement. (not yet implemented)"""
    click.echo("Not yet implemented.")


@cli.command()
def run():
    """Run the full pipeline. (not yet implemented)"""
    click.echo("Not yet implemented.")
