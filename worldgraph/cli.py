from pathlib import Path

import click

from worldgraph.extract import run_extraction


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
def cluster():
    """Stage 2: Embed and cluster similar relations. (not yet implemented)"""
    click.echo("Not yet implemented.")


@cli.command()
def match():
    """Stage 3: Structural matching across article subgraphs. (not yet implemented)"""
    click.echo("Not yet implemented.")


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
