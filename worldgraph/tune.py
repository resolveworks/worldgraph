"""Parameter sweep over similarity threshold.

Embeds once, then runs propagation+merge for each threshold value in-memory.
"""

from pathlib import Path

import click

from worldgraph.eval import load_ground_truth, score_graphs
from worldgraph.match import load_graphs, prepare_embeddings, run_match_merge

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
MAX_ITER = 30
EPSILON = 1e-4


def run_sweep(input_path: Path, ground_truth_path: Path) -> None:
    graphs, entity_occurrences, edge_articles = load_graphs(input_path)
    click.echo(f"Loaded {len(graphs)} graphs from {input_path}")

    click.echo("Embedding entity names and relation phrases (once)...")
    name_embeddings, relation_embeddings, functionality = prepare_embeddings(graphs)

    name_to_canonical = load_ground_truth(ground_truth_path)

    click.echo(f"Sweeping {len(THRESHOLDS)} threshold values...\n")

    results = []
    for threshold in THRESHOLDS:
        merged_graphs, merged_occ, _ = run_match_merge(
            graphs, entity_occurrences, edge_articles,
            name_embeddings, relation_embeddings, functionality,
            threshold=threshold,
            max_iter=MAX_ITER,
            epsilon=EPSILON,
        )
        precision, recall, f1 = score_graphs(merged_graphs, merged_occ, name_to_canonical)
        results.append((threshold, precision, recall, f1))

    results.sort(key=lambda r: r[3], reverse=True)

    header = f"{'threshold':>10}  {'precision':>9}  {'recall':>6}  {'f1':>6}"
    click.echo(header)
    click.echo("-" * len(header))

    best_f1 = results[0][3]
    for threshold, precision, recall, f1 in results:
        marker = " <-- best" if f1 == best_f1 else ""
        click.echo(
            f"{threshold:>10.2f}  {precision:>9.1%}  {recall:>6.1%}  {f1:>6.1%}{marker}"
        )
