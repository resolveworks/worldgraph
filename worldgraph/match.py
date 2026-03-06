"""Stage 2: Entity alignment via PARIS-style similarity propagation.

Entity names are stored directly on nodes.  Name similarity seeds the
confidence dict before the iteration loop, so structural evidence
propagates from iteration 1.  Exponential sum aggregation, threshold,
merge via union-find.
"""

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
from dotenv import load_dotenv
from worldgraph.constants import RELATION_TEMPLATE
from worldgraph.embed import Embedder
from worldgraph.graph import (
    Graph,
    load_graph,
    save_graph,
)
from worldgraph.names import build_idf, soft_tfidf

load_dotenv()


class Functionality(NamedTuple):
    forward: float
    inverse: float


class Neighbor(NamedTuple):
    """An entry in a node's weighted adjacency list."""

    entity_id: str
    relation: str
    func_weight: float


# Type aliases for the main data structures flowing through the pipeline.
Confidence = dict[tuple[str, str], float]
MatchGroup = set[str]


class UnionFind:
    def __init__(self):
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Embeddings and relation functionality
# ---------------------------------------------------------------------------


def compute_functionality(
    graphs: list[Graph],
    relation_embeddings: dict[str, np.ndarray],
    threshold: float = 0.8,
) -> dict[str, Functionality]:
    """Compute functionality and inverse functionality for each relation phrase.

    Functionality ≈ 1 / avg_out_degree: for a given source name, how many
    distinct target names does it map to via this relation pool? High means
    the relation uniquely determines the target — strong forward evidence.

    Inverse functionality ≈ 1 / avg_in_degree: for a given target name, how
    many distinct source names map to it via this relation pool? High means
    the relation uniquely determines the source — strong backward evidence.

    Entity names (not IDs) are used so that the same entity mentioned across
    multiple graphs pools its statistics. For each relation phrase r, edges
    whose phrase r' satisfies dot(r, r') >= threshold are pooled
    together — the same threshold used in similarity propagation.

    Returns dict from phrase to Functionality(forward, inverse).
    """
    all_relations = list(relation_embeddings)

    # Precompute which relations are similar to each relation
    similar: dict[str, list[str]] = {rel: [] for rel in all_relations}
    for rel in all_relations:
        for other_rel in all_relations:
            if (
                float(np.dot(relation_embeddings[rel], relation_embeddings[other_rel]))
                >= threshold
            ):
                similar[rel].append(other_rel)

    # Collect all (source_name, target_name) pairs per relation phrase
    phrase_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for graph in graphs:
        for edge in graph.edges:
            source_name = graph.nodes[edge.source].name
            target_name = graph.nodes[edge.target].name
            phrase_pairs[edge.relation].append((source_name, target_name))

    result: dict[str, Functionality] = {}
    for rel in all_relations:
        pooled = [
            pair
            for other_rel in similar[rel]
            for pair in phrase_pairs.get(other_rel, [])
        ]
        if not pooled:
            result[rel] = Functionality(1.0, 1.0)
            continue
        targets_per_source: dict[str, set[str]] = defaultdict(set)
        sources_per_target: dict[str, set[str]] = defaultdict(set)
        for source_name, target_name in pooled:
            targets_per_source[source_name].add(target_name)
            sources_per_target[target_name].add(source_name)
        avg_out_degree = sum(
            len(targets) for targets in targets_per_source.values()
        ) / len(targets_per_source)
        avg_in_degree = sum(
            len(sources) for sources in sources_per_target.values()
        ) / len(sources_per_target)
        result[rel] = Functionality(1.0 / avg_out_degree, 1.0 / avg_in_degree)

    return result


# ---------------------------------------------------------------------------
# Similarity propagation
# ---------------------------------------------------------------------------


def _build_weighted_adjacency(
    graph: Graph,
    functionality: dict[str, Functionality],
) -> dict[str, list[Neighbor]]:
    """Build per-entity adjacency list with direction-appropriate functionality.

    PARIS semantics: the functionality weight measures "if my neighbor matches,
    how strong is the evidence that I match?"

    For edge source --r--> target:
      - source uses inverse functionality: "given the target, how unique is the
        source?"  If fun⁻¹(r) ≈ 1, a target match strongly implies a source match.
      - target uses forward functionality: "given the source, how unique is the
        target?"  If fun(r) ≈ 1, a source match strongly implies a target match.
    """
    default = Functionality(1.0, 1.0)
    adjacency: dict[str, list[Neighbor]] = defaultdict(list)
    for edge in graph.edges:
        func = functionality.get(edge.relation, default)
        adjacency[edge.source].append(
            Neighbor(edge.target, edge.relation, func.inverse)
        )
        adjacency[edge.target].append(
            Neighbor(edge.source, edge.relation, func.forward)
        )
    return adjacency


def build_unified_graph(graphs: list[Graph]) -> Graph:
    """Combine N article graphs into one. Node IDs are UUIDs — unique across graphs."""
    unified = Graph()
    for graph in graphs:
        unified.nodes.update(graph.nodes)
        unified.edges.extend(graph.edges)
    return unified


def propagate_similarity(
    graph: Graph,
    idf: dict[str, float],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    max_iter: int = 30,
    epsilon: float = 1e-4,
    exp_lambda: float = 1.0,
) -> Confidence:
    """Run similarity propagation on a single unified graph.

    Compares entity pairs from different source graphs (based on
    node.graph_id).  Name similarity seeds the confidence dict before
    the iteration loop.  Entity-entity confidence is updated iteratively
    using double-buffering (each iteration reads from the previous
    iteration's values).

    Relation similarity and neighbor confidence are continuous
    multipliers — no hard gates.  Each propagation path contributes:

        rel_sim(r_a, r_b) × min(func_a, func_b) × neighbor_confidence

    Returns confidence: (entity_id_a, entity_id_b) -> float in [0, 1].
    Both orderings (a,b) and (b,a) are stored for convenient lookup.
    """
    adjacency = _build_weighted_adjacency(graph, functionality)

    # Precompute pairwise relation similarities (continuous, not gated)
    all_relations = {edge.relation for edge in graph.edges}
    rel_sim: dict[tuple[str, str], float] = {}
    for rel_a in all_relations:
        embedding_a = relation_embeddings.get(rel_a)
        if embedding_a is None:
            continue
        for rel_b in all_relations:
            embedding_b = relation_embeddings.get(rel_b)
            if embedding_b is None:
                continue
            rel_sim[(rel_a, rel_b)] = max(0.0, float(np.dot(embedding_a, embedding_b)))

    entity_ids = list(graph.nodes.keys())

    # Seed confidence from name similarity before the iteration loop.
    confidence: Confidence = {}
    pairs: list[tuple[str, str]] = []
    for i, id_a in enumerate(entity_ids):
        for id_b in entity_ids[i + 1 :]:
            if graph.nodes[id_a].graph_id == graph.nodes[id_b].graph_id:
                continue
            name_sim = max(
                0.0,
                soft_tfidf(graph.nodes[id_a].name, graph.nodes[id_b].name, idf),
            )
            confidence[(id_a, id_b)] = name_sim
            confidence[(id_b, id_a)] = name_sim
            pairs.append((id_a, id_b))

    for _ in range(max_iter):
        prev = dict(confidence)
        changed = False

        for id_a, id_b in pairs:
            strength_sum = 0.0

            for neighbor_a in adjacency.get(id_a, []):
                for neighbor_b in adjacency.get(id_b, []):
                    rs = rel_sim.get((neighbor_a.relation, neighbor_b.relation), 0.0)
                    if rs <= 0.0:
                        continue
                    neighbor_confidence = prev.get(
                        (neighbor_a.entity_id, neighbor_b.entity_id), 0.0
                    )
                    if neighbor_confidence <= 0.0:
                        continue
                    weight = min(neighbor_a.func_weight, neighbor_b.func_weight)
                    strength_sum += rs * weight * neighbor_confidence

            combined = (
                1.0 - math.exp(-exp_lambda * strength_sum) if strength_sum > 0 else 0.0
            )

            old = prev[(id_a, id_b)]
            if combined > old + epsilon:
                confidence[(id_a, id_b)] = combined
                confidence[(id_b, id_a)] = combined
                changed = True

        if not changed:
            break

    return confidence


# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------


def match_graphs(
    graphs: list[Graph],
    embedder: Embedder,
    rel_cluster_threshold: float = 0.8,
    **propagate_kwargs,
) -> Confidence:
    """Core matching pipeline: graphs → confidence scores.

    Builds unified graph, computes IDF / relation embeddings / functionality,
    and runs similarity propagation. Returns the confidence dict.
    """
    unified = build_unified_graph(graphs)

    all_names = [node.name for graph in graphs for node in graph.nodes.values()]
    all_relations = sorted({edge.relation for graph in graphs for edge in graph.edges})

    idf = build_idf(all_names)
    relation_embeddings = embedder.embed(all_relations, template=RELATION_TEMPLATE)
    functionality = compute_functionality(
        graphs, relation_embeddings, rel_cluster_threshold
    )

    return propagate_similarity(
        unified,
        idf,
        relation_embeddings,
        functionality,
        **propagate_kwargs,
    )


def build_match_groups(
    graphs: list[Graph],
    confidence: Confidence,
    threshold: float = 0.8,
) -> list[MatchGroup]:
    """Build match groups from confidence scores via union-find.

    Returns list of sets, each containing matched entity IDs (groups of size > 1).
    """
    uf = UnionFind()
    for (id_a, id_b), score in confidence.items():
        if score >= threshold:
            uf.union(id_a, id_b)

    unified = build_unified_graph(graphs)
    groups: dict[str, list[str]] = defaultdict(list)
    for entity_id in unified.nodes:
        groups[uf.find(entity_id)].append(entity_id)
    return [set(members) for members in groups.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_matching(
    graph_files: list[Path],
    output_path: Path,
    relation_threshold: float,
    match_threshold: float,
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> None:
    """Load graphs, run matching pipeline, save results."""
    graphs = [load_graph(path) for path in graph_files]
    click.echo(f"Loaded {len(graphs)} graphs")
    for graph in graphs:
        click.echo(
            f"  {graph.id}: {len(graph.nodes)} entities, {len(graph.edges)} edges"
        )

    embedder = Embedder(os.environ["EMBEDDING_MODEL"])

    confidence = match_graphs(
        graphs,
        embedder,
        rel_cluster_threshold=relation_threshold,
        max_iter=max_iter,
        epsilon=epsilon,
    )

    match_groups = build_match_groups(graphs, confidence, match_threshold)

    unified = build_unified_graph(graphs)
    save_graph(unified, output_path, [list(group) for group in match_groups])

    click.echo(f"\n{len(match_groups)} match groups:")
    for members in match_groups:
        names = {unified.nodes[eid].name for eid in members}
        click.echo(f"  {' / '.join(sorted(names))}")

    click.echo(f"\nWrote {output_path}")
