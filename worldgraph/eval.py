"""Evaluate matching output against ground truth canonical entity IDs.

For each merged entity in the output graphs, all occurrence names should
resolve to the same canonical ID (precision). For each canonical ID, all
occurrences across articles should land in the same merged entity (recall).

Usage:
    worldgraph eval                          # evaluate data/graphs_1.json
    worldgraph eval -i data/graphs_2.json
"""

import json
from collections import defaultdict
from pathlib import Path

import click


def load_ground_truth(path: Path) -> dict[str, str]:
    """Return name -> canonical_id mapping."""
    with open(path) as f:
        return json.load(f)["name_to_canonical"]


def evaluate(graphs_path: Path, ground_truth_path: Path, verbose: bool) -> None:
    with open(graphs_path) as f:
        data = json.load(f)

    name_to_canonical = load_ground_truth(ground_truth_path)

    # --- Build evaluation structures ---

    # merged_entity_id -> set of canonical_ids found in its occurrences
    entity_canonicals: dict[str, set[str]] = defaultdict(set)
    # merged_entity_id -> list of occurrence names (for reporting)
    entity_names: dict[str, list[str]] = defaultdict(list)
    # canonical_id -> set of merged_entity_ids that contain it
    canonical_to_entities: dict[str, set[str]] = defaultdict(set)

    for g in data["graphs"]:
        for e in g["entities"]:
            eid = e["id"]
            for occ in e["occurrences"]:
                name = occ["name"]
                entity_names[eid].append(name)
                canonical = name_to_canonical.get(name)
                if canonical is not None:
                    entity_canonicals[eid].add(canonical)
                    canonical_to_entities[canonical].add(eid)

    # --- Precision: false merges ---
    # A merged entity is a false merge if its occurrences span >1 canonical ID
    false_merges = {
        eid: canonicals
        for eid, canonicals in entity_canonicals.items()
        if len(canonicals) > 1
    }

    # --- Recall: missed merges ---
    # A canonical ID is a missed merge if its occurrences ended up in >1 merged entity
    missed_merges = {
        canonical: entities
        for canonical, entities in canonical_to_entities.items()
        if len(entities) > 1
    }

    # --- Summary counts ---
    # Only count entities that have at least one known-canonical occurrence
    known_entities = {eid for eid, c in entity_canonicals.items() if c}
    correct = len(known_entities) - len(false_merges)

    click.echo(f"Evaluated: {graphs_path}")
    click.echo(f"Ground truth: {ground_truth_path}")
    click.echo()

    total_canonicals = len(canonical_to_entities)
    fully_merged = sum(1 for c in canonical_to_entities.values() if len(c) == 1)
    click.echo(f"Recall:    {fully_merged}/{total_canonicals} canonicals fully merged"
               f"  ({100*fully_merged/total_canonicals:.0f}%)" if total_canonicals else "Recall: n/a")

    total_known = len(known_entities)
    click.echo(f"Precision: {correct}/{total_known} merged entities are clean"
               f"  ({100*correct/total_known:.0f}%)" if total_known else "Precision: n/a")

    if false_merges:
        click.echo(f"\n{len(false_merges)} FALSE MERGES (different entities merged together):")
        for eid, canonicals in false_merges.items():
            names = sorted(set(entity_names[eid]))
            click.echo(f"  [{', '.join(sorted(canonicals))}]")
            if verbose:
                click.echo(f"    names: {', '.join(names)}")
    else:
        click.echo("\nNo false merges.")

    if missed_merges:
        click.echo(f"\n{len(missed_merges)} MISSED MERGES (same entity not merged):")
        for canonical, entities in missed_merges.items():
            if verbose:
                for eid in entities:
                    names = sorted(set(entity_names[eid]))
                    click.echo(f"  {canonical}: {', '.join(names)}")
            else:
                click.echo(f"  {canonical} ({len(entities)} fragments)")
    else:
        click.echo("No missed merges.")
