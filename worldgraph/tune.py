"""Parameter sweep over threshold × evidence_scale.

Embeds once, then runs match+merge for each combination in-memory.
"""

from pathlib import Path

import click

from worldgraph.eval import load_ground_truth, score_graphs
from worldgraph.match import load_graphs, prepare_embeddings, run_match_merge

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
EVIDENCE_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]
REL_FLOOR = 0.8


def run_sweep(
    input_path: Path,
    ground_truth_path: Path,
) -> None:
    graphs, entity_occurrences, edge_articles = load_graphs(input_path)
    click.echo(f"Loaded {len(graphs)} graphs from {input_path}")

    click.echo("Embedding entity names and relation phrases (once)...")
    name_embeddings, relation_embeddings, relation_specificities = prepare_embeddings(graphs)

    name_to_canonical = load_ground_truth(ground_truth_path)

    n_combos = len(THRESHOLDS) * len(EVIDENCE_SCALES)
    click.echo(f"Sweeping {n_combos} parameter combinations...\n")

    results = []
    for threshold in THRESHOLDS:
        for evidence_scale in EVIDENCE_SCALES:
            merged_graphs, merged_occ, _ = run_match_merge(
                graphs, entity_occurrences, edge_articles,
                name_embeddings, relation_embeddings, relation_specificities,
                threshold=threshold,
                rel_floor=REL_FLOOR,
                evidence_scale=evidence_scale,
            )
            precision, recall, f1 = score_graphs(merged_graphs, merged_occ, name_to_canonical)
            results.append((threshold, evidence_scale, precision, recall, f1))

    results.sort(key=lambda r: r[4], reverse=True)

    header = f"{'threshold':>10}  {'ev_scale':>8}  {'precision':>9}  {'recall':>6}  {'f1':>6}"
    click.echo(header)
    click.echo("-" * len(header))

    best_f1 = results[0][4]
    for threshold, evidence_scale, precision, recall, f1 in results:
        marker = " <-- best" if f1 == best_f1 else ""
        click.echo(
            f"{threshold:>10.2f}  {evidence_scale:>8.1f}  "
            f"{precision:>9.1%}  {recall:>6.1%}  {f1:>6.1%}{marker}"
        )
