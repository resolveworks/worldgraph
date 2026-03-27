"""Stage 2: Entity alignment via PARIS-style similarity propagation.

Entity names are stored as lists on nodes (multi-label).  Name similarity
seeds the confidence dict before the iteration loop using the max over all
name pairs, so structural evidence propagates from iteration 1.  Relation
similarity is treated as binary
via a single threshold that defines equivalence classes over free-text
relation phrases — above threshold = same relation, below = different.
This threshold is used consistently for functionality pooling, positive
propagation gating, and negative evidence.  Exponential sum aggregation,
threshold, merge via union-find.
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
            source_name = graph.nodes[edge.source].names[0]
            target_name = graph.nodes[edge.target].names[0]
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


def compute_negative_factor(
    id_a: str,
    id_b: str,
    forward_adj: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    confidence: Confidence,
    alpha: float = 0.3,
    floor: float = 0.5,
    *,
    rel_threshold: float,
) -> float:
    """Compute dampened negative factor for an entity pair.

    For each neighbor y of a (via relation r), check whether y matches any
    neighbor y' of b (via a similar relation r'). If no match is found and
    the relation is functional, penalize the pair.

    Both directions are checked independently; the more charitable (higher)
    factor is used.  This reflects news graph reality: articles cover
    different aspects of the same entity, so missing neighbors on one side
    is common and should not compound penalties from both directions.

    ``rel_threshold`` is the same relation equivalence threshold used by
    positive propagation and functionality pooling.

    Returns a value in (0, 1] that multiplies the positive confidence.
    """
    neg_a = _one_sided_negative(
        id_a,
        id_b,
        forward_adj,
        rel_sim,
        confidence,
        alpha,
        floor,
        rel_threshold=rel_threshold,
    )
    neg_b = _one_sided_negative(
        id_b,
        id_a,
        forward_adj,
        rel_sim,
        confidence,
        alpha,
        floor,
        rel_threshold=rel_threshold,
    )
    return max(neg_a, neg_b)


def _one_sided_negative(
    id_a: str,
    id_b: str,
    forward_adj: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    confidence: Confidence,
    alpha: float,
    floor: float,
    *,
    rel_threshold: float,
) -> float:
    """Negative factor from a's perspective.

    Computes a functionality-weighted average mismatch across a's neighbors,
    then converts to a multiplicative penalty.  The weighted average
    naturally normalizes for the number of neighbors: an entity with
    5 neighbors, 4 matching and 1 not, gets a mild penalty (20% mismatch),
    while an entity with 1 non-matching neighbor gets a strong one (100%).

    Relation similarity is treated as binary (matching or not) using
    ``rel_threshold`` — the same threshold used by positive propagation
    and functionality pooling.
    """
    neighbors_a = forward_adj.get(id_a, [])
    if not neighbors_a:
        return 1.0

    total_weight = 0.0
    weighted_mismatch = 0.0
    for neighbor_a in neighbors_a:
        match_prob = 0.0
        for neighbor_b in forward_adj.get(id_b, []):
            rs = rel_sim.get((neighbor_a.relation, neighbor_b.relation), 0.0)
            if rs < rel_threshold:
                continue
            nbr_conf = confidence.get((neighbor_a.entity_id, neighbor_b.entity_id), 0.0)
            match_prob += nbr_conf
        match_prob = min(match_prob, 1.0)
        total_weight += neighbor_a.func_weight
        weighted_mismatch += neighbor_a.func_weight * (1.0 - match_prob)

    if total_weight == 0.0:
        return 1.0

    avg_mismatch = weighted_mismatch / total_weight
    return max(1.0 - alpha * avg_mismatch, floor)


def build_unified_graph(graphs: list[Graph]) -> Graph:
    """Combine N article graphs into one. Node IDs are UUIDs — unique across graphs."""
    unified = Graph()
    for graph in graphs:
        unified.nodes.update(graph.nodes)
        unified.edges.extend(graph.edges)
    return unified


def propagate_positive(
    adjacency: dict[str, list[Neighbor]],
    pairs: list[tuple[str, str]],
    positive_base: Confidence,
    *,
    rel_sim: dict[tuple[str, str], float],
    rel_threshold: float,
    max_iter: int,
    epsilon: float,
    exp_lambda: float,
) -> Confidence:
    """Run the monotone non-decreasing positive fixpoint loop.

    Updates ``positive_base`` in place and returns it.  Each iteration
    reads from the previous snapshot (double-buffering) and applies the
    monotone max rule: new value = max(structural_update, old_value).
    """
    for _ in range(max_iter):
        prev_base = dict(positive_base)
        changed = False

        for id_a, id_b in pairs:
            strength_sum = 0.0

            for neighbor_a in adjacency.get(id_a, []):
                for neighbor_b in adjacency.get(id_b, []):
                    rs = rel_sim.get((neighbor_a.relation, neighbor_b.relation), 0.0)
                    if rs < rel_threshold:
                        continue
                    neighbor_confidence = prev_base.get(
                        (neighbor_a.entity_id, neighbor_b.entity_id), 0.0
                    )
                    if neighbor_confidence <= 0.0:
                        continue
                    weight = min(neighbor_a.func_weight, neighbor_b.func_weight)
                    strength_sum += weight * neighbor_confidence

            positive = (
                1.0 - math.exp(-exp_lambda * strength_sum) if strength_sum > 0 else 0.0
            )

            old_base = prev_base[(id_a, id_b)]
            base = max(positive, old_base)
            positive_base[(id_a, id_b)] = base
            positive_base[(id_b, id_a)] = base

            if abs(base - old_base) > epsilon:
                changed = True

        if not changed:
            break

    return positive_base


def apply_negative(
    positive_base: Confidence,
    pairs: list[tuple[str, str]],
    forward_adj: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    name_seed: Confidence,
    *,
    neg_alpha: float,
    neg_floor: float,
    neg_gate: float,
    rel_threshold: float,
) -> Confidence:
    """Apply negative dampening as a single post-convergence pass.

    The negative factor uses ``name_seed`` (fixed name similarity) to
    check whether neighbors match, preventing circular reinforcement.
    Returns a new confidence dict with dampened values.
    """
    confidence = dict(positive_base)
    for id_a, id_b in pairs:
        base = positive_base[(id_a, id_b)]
        if base > neg_gate:
            neg = compute_negative_factor(
                id_a,
                id_b,
                forward_adj,
                rel_sim,
                name_seed,
                alpha=neg_alpha,
                floor=neg_floor,
                rel_threshold=rel_threshold,
            )
            combined = base * neg
            confidence[(id_a, id_b)] = combined
            confidence[(id_b, id_a)] = combined
    return confidence


def _build_rel_sim(
    graph: Graph,
    relation_embeddings: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Precompute pairwise relation similarities for all relations in graph."""
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
    return rel_sim


def _build_epoch_adjacency(
    graph: Graph,
    functionality: dict[str, Functionality],
    uf: UnionFind,
) -> tuple[dict[str, list[Neighbor]], dict[str, list[Neighbor]]]:
    """Build adjacency lists using union-find canonical reps.

    Merged entities' neighborhoods are unioned: each edge contributes
    neighbors keyed by the canonical rep of both endpoints.  Duplicate
    entries (same canonical neighbor + same relation) are deduplicated
    to prevent inflated evidence from merged entities having multiple
    copies of structurally identical edges.
    """
    default = Functionality(1.0, 1.0)
    # Collect unique (entity, neighbor, relation) triples per direction
    adj_seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    fwd_seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    adjacency: dict[str, list[Neighbor]] = defaultdict(list)
    forward_adj: dict[str, list[Neighbor]] = defaultdict(list)
    for edge in graph.edges:
        func = functionality.get(edge.relation, default)
        src = uf.find(edge.source)
        tgt = uf.find(edge.target)
        # Weighted adjacency (for positive propagation)
        key_src = (tgt, edge.relation)
        if key_src not in adj_seen[src]:
            adj_seen[src].add(key_src)
            adjacency[src].append(Neighbor(tgt, edge.relation, func.inverse))
        key_tgt = (src, edge.relation)
        if key_tgt not in adj_seen[tgt]:
            adj_seen[tgt].add(key_tgt)
            adjacency[tgt].append(Neighbor(src, edge.relation, func.forward))
        # Forward adjacency (for negative evidence)
        if key_src not in fwd_seen[src]:
            fwd_seen[src].add(key_src)
            forward_adj[src].append(Neighbor(tgt, edge.relation, func.forward))
        if key_tgt not in fwd_seen[tgt]:
            fwd_seen[tgt].add(key_tgt)
            forward_adj[tgt].append(Neighbor(src, edge.relation, func.inverse))
    return adjacency, forward_adj


def _build_epoch_pairs(
    graph: Graph,
    uf: UnionFind,
) -> list[tuple[str, str]]:
    """Build cross-graph entity pairs between canonical reps.

    Maps original entity IDs through uf.find(), deduplicates, and
    skips pairs where all members share a single source graph.
    """
    # Collect which graph_ids each canonical rep covers
    canon_graphs: dict[str, set[str]] = defaultdict(set)
    for eid, node in graph.nodes.items():
        canon_graphs[uf.find(eid)].add(node.graph_id)

    canons = sorted(canon_graphs.keys())
    pairs: list[tuple[str, str]] = []
    for i, ca in enumerate(canons):
        for cb in canons[i + 1 :]:
            # Skip if all members of both groups are from the same graph
            if len(canon_graphs[ca] | canon_graphs[cb]) == 1:
                continue
            pairs.append((ca, cb))
    return pairs


def _seed_epoch_confidence(
    graph: Graph,
    idf: dict[str, float],
    uf: UnionFind,
    pairs: list[tuple[str, str]],
    prev_confidence: Confidence | None = None,
) -> tuple[Confidence, Confidence]:
    """Seed confidence for an epoch using max name similarity across members.

    For each canonical pair (ca, cb), the name seed is the maximum
    soft-TFIDF score across all name pairs from all members of both
    groups.  The full seed additionally carries forward previous epoch
    confidence.

    Returns (full_seed, name_seed) — the full seed is used for positive
    propagation, while the name-only seed is used for negative evidence
    to prevent circular reinforcement across epochs.
    """
    # Build member lists per canonical rep
    members: dict[str, list[str]] = defaultdict(list)
    for eid in graph.nodes:
        members[uf.find(eid)].append(eid)

    full_seed: Confidence = {}
    name_seed: Confidence = {}
    for ca, cb in pairs:
        # Max name similarity across all member-pair name comparisons
        best_name = 0.0
        for ma in members[ca]:
            for mb in members[cb]:
                if graph.nodes[ma].graph_id == graph.nodes[mb].graph_id:
                    continue
                for na in graph.nodes[ma].names:
                    for nb in graph.nodes[mb].names:
                        best_name = max(best_name, soft_tfidf(na, nb, idf))
        best_name = max(0.0, best_name)

        name_seed[(ca, cb)] = best_name
        name_seed[(cb, ca)] = best_name

        # Carry forward previous confidence between any member pairs
        best_prev = 0.0
        if prev_confidence:
            for ma in members[ca]:
                for mb in members[cb]:
                    best_prev = max(
                        best_prev,
                        prev_confidence.get((ma, mb), 0.0),
                    )

        seed = max(best_name, best_prev)
        full_seed[(ca, cb)] = seed
        full_seed[(cb, ca)] = seed

    return full_seed, name_seed


def propagate_similarity(
    graph: Graph,
    idf: dict[str, float],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    rel_threshold: float = 0.8,
    max_iter: int = 30,
    epsilon: float = 1e-4,
    exp_lambda: float = 1.0,
    neg_alpha: float = 0.3,
    neg_floor: float = 0.5,
    neg_gate: float = 0.3,
    merge_threshold: float = 0.9,
    max_epochs: int = 5,
) -> tuple[Confidence, UnionFind]:
    """Run epoch-based similarity propagation with progressive merging.

    Each epoch runs the positive fixpoint loop to convergence, applies
    negative dampening, then commits high-confidence merges via union-find.
    Merged entities' neighborhoods are unioned for subsequent epochs,
    allowing evidence from transitively-matched entities to compound.

    With default ``merge_threshold=0.9`` and ``max_epochs=5``, pairs
    scoring below 0.9 never trigger progressive merges, so the epoch
    loop exits after one epoch — reproducing the previous non-epoch
    behavior.

    Returns (confidence, union_find) where confidence maps canonical-rep
    pairs to scores and union_find tracks all committed merges.
    """
    rel_sim = _build_rel_sim(graph, relation_embeddings)
    uf = UnionFind()

    # Initialize all entities in the union-find
    for eid in graph.nodes:
        uf.find(eid)

    confidence: Confidence = {}
    # Track the best score seen for each original entity pair across all
    # epochs.  Enriched neighborhoods in later epochs can strengthen
    # negative evidence, but earlier positive evidence should not be lost.
    best_confidence: Confidence = {}
    prev_epoch_confidence: Confidence | None = None

    for _epoch in range(max_epochs):
        adjacency, forward_adj = _build_epoch_adjacency(graph, functionality, uf)
        pairs = _build_epoch_pairs(graph, uf)

        if not pairs:
            break

        confidence, name_seed = _seed_epoch_confidence(
            graph, idf, uf, pairs, prev_epoch_confidence
        )

        positive_base = propagate_positive(
            adjacency,
            pairs,
            dict(confidence),
            rel_sim=rel_sim,
            rel_threshold=rel_threshold,
            max_iter=max_iter,
            epsilon=epsilon,
            exp_lambda=exp_lambda,
        )

        confidence = apply_negative(
            positive_base,
            pairs,
            forward_adj,
            rel_sim,
            name_seed,
            neg_alpha=neg_alpha,
            neg_floor=neg_floor,
            neg_gate=neg_gate,
            rel_threshold=rel_threshold,
        )

        # Expand this epoch's canonical-rep scores to original entity
        # pairs and merge into best_confidence.
        members_now: dict[str, list[str]] = defaultdict(list)
        for eid in graph.nodes:
            members_now[uf.find(eid)].append(eid)

        for (ca, cb), score in confidence.items():
            if ca == cb:
                continue
            for ma in members_now.get(ca, [ca]):
                for mb in members_now.get(cb, [cb]):
                    if graph.nodes[ma].graph_id == graph.nodes[mb].graph_id:
                        continue
                    old = best_confidence.get((ma, mb), -1.0)
                    new = max(old, score)
                    best_confidence[(ma, mb)] = new
                    best_confidence[(mb, ma)] = new

        # Find new merges above merge_threshold
        new_merges = []
        for ca, cb in pairs:
            if confidence.get((ca, cb), 0.0) >= merge_threshold:
                if uf.find(ca) != uf.find(cb):
                    new_merges.append((ca, cb))

        if not new_merges:
            break

        for ca, cb in new_merges:
            uf.union(ca, cb)

        prev_epoch_confidence = confidence

    # Set confidence=1.0 for all pairs within the same UF group
    # (they were merged with high confidence during epochs).
    members: dict[str, list[str]] = defaultdict(list)
    for eid in graph.nodes:
        members[uf.find(eid)].append(eid)

    for group_members in members.values():
        if len(group_members) < 2:
            continue
        for i, ma in enumerate(group_members):
            for mb in group_members[i + 1 :]:
                if graph.nodes[ma].graph_id == graph.nodes[mb].graph_id:
                    continue
                best_confidence[(ma, mb)] = 1.0
                best_confidence[(mb, ma)] = 1.0

    return best_confidence, uf


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
    and runs similarity propagation.  ``rel_cluster_threshold`` is the single
    relation equivalence threshold: relation pairs with embedding similarity
    above this value are treated as the same relation for functionality
    pooling, positive propagation gating, and negative evidence.

    Returns the confidence dict.
    """
    unified = build_unified_graph(graphs)

    all_names = [
        name for graph in graphs for node in graph.nodes.values() for name in node.names
    ]
    all_relations = sorted({edge.relation for graph in graphs for edge in graph.edges})

    idf = build_idf(all_names)
    relation_embeddings = embedder.embed(all_relations, template=RELATION_TEMPLATE)
    functionality = compute_functionality(
        graphs, relation_embeddings, rel_cluster_threshold
    )

    confidence, _uf = propagate_similarity(
        unified,
        idf,
        relation_embeddings,
        functionality,
        rel_threshold=rel_cluster_threshold,
        **propagate_kwargs,
    )
    return confidence


def build_match_groups(
    graphs: list[Graph],
    confidence: Confidence,
    threshold: float = 0.8,
) -> tuple[list[MatchGroup], Graph]:
    """Build match groups from confidence scores via union-find.

    Returns (match_groups, unified_graph) where match_groups is a list of sets,
    each containing matched entity IDs (groups of size > 1).
    """
    uf = UnionFind()
    for (id_a, id_b), score in confidence.items():
        if score >= threshold:
            uf.union(id_a, id_b)

    unified = build_unified_graph(graphs)
    groups: dict[str, list[str]] = defaultdict(list)
    for entity_id in unified.nodes:
        groups[uf.find(entity_id)].append(entity_id)
    return [set(members) for members in groups.values() if len(members) > 1], unified


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
    merge_threshold: float = 0.9,
    max_epochs: int = 5,
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
        merge_threshold=merge_threshold,
        max_epochs=max_epochs,
    )

    match_groups, unified = build_match_groups(graphs, confidence, match_threshold)
    save_graph(unified, output_path, [list(group) for group in match_groups])

    click.echo(f"\n{len(match_groups)} match groups:")
    for members in match_groups:
        names = {n for eid in members for n in unified.nodes[eid].names}
        click.echo(f"  {' / '.join(sorted(names))}")

    click.echo(f"\nWrote {output_path}")
