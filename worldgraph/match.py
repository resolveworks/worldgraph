"""Stage 2: Entity alignment via PARIS-style similarity propagation.

Entity names are stored directly on nodes.  Name similarity seeds the
confidence dict before the iteration loop, so structural evidence
propagates from iteration 1.  Relation similarity is treated as binary
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


def _build_forward_adjacency(
    graph: Graph,
    functionality: dict[str, Functionality],
) -> dict[str, list[Neighbor]]:
    """Build per-entity forward adjacency for negative evidence.

    For negative evidence, we need forward functionality: "given the source,
    how many targets does this relation map to?" If the answer is one (high
    forward functionality) and the target doesn't match, that's damning.

    For each edge source --r--> target, the source gets a neighbor entry
    with forward functionality. Both directions are included symmetrically:
    the target also gets a neighbor entry (with inverse functionality as the
    "forward" direction from target's perspective).
    """
    default = Functionality(1.0, 1.0)
    adjacency: dict[str, list[Neighbor]] = defaultdict(list)
    for edge in graph.edges:
        func = functionality.get(edge.relation, default)
        adjacency[edge.source].append(
            Neighbor(edge.target, edge.relation, func.forward)
        )
        adjacency[edge.target].append(
            Neighbor(edge.source, edge.relation, func.inverse)
        )
    return adjacency


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
            if neighbor_a.entity_id == neighbor_b.entity_id:
                nbr_conf = 1.0
            else:
                nbr_conf = confidence.get(
                    (neighbor_a.entity_id, neighbor_b.entity_id), 0.0
                )
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


def _build_canonical_adjacency(
    adjacency: dict[str, list[Neighbor]],
    uf: "UnionFind",
) -> dict[str, list[Neighbor]]:
    """Build deduplicated adjacency using union-find canonical reps.

    Maps each node to its canonical rep via ``uf.find()``, then deduplicates:
    for each (canon_node, canon_neighbor, relation) triple, keeps only the
    entry with the highest func_weight.
    """
    # Collect best weight per (canon_node, canon_neighbor, relation)
    best: dict[tuple[str, str, str], float] = {}
    for node_id, neighbors in adjacency.items():
        canon_node = uf.find(node_id)
        for nbr in neighbors:
            canon_nbr = uf.find(nbr.entity_id)
            if canon_nbr == canon_node:
                continue
            key = (canon_node, canon_nbr, nbr.relation)
            if nbr.func_weight > best.get(key, 0.0):
                best[key] = nbr.func_weight

    result: dict[str, list[Neighbor]] = defaultdict(list)
    for (canon_node, canon_nbr, relation), weight in best.items():
        result[canon_node].append(Neighbor(canon_nbr, relation, weight))
    return result


def propagate_positive(
    pairs: list[tuple[str, str]],
    positive_base: Confidence,
    adjacency: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    rel_threshold: float = 0.8,
    max_iter: int = 30,
    epsilon: float = 1e-4,
    exp_lambda: float = 1.0,
) -> Confidence:
    """Run the monotone positive fixpoint loop.

    Iterates until convergence or ``max_iter``, updating ``positive_base``
    in place.  Each propagation path contributes:

        min(func_a, func_b) × neighbor_positive_base

    Returns the converged positive_base (same object, mutated).
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
                    # Same canonical neighbor = confirmed shared neighbor.
                    if neighbor_a.entity_id == neighbor_b.entity_id:
                        neighbor_confidence = 1.0
                    else:
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
    pairs: list[tuple[str, str]],
    positive_base: Confidence,
    forward_adj: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    name_seed: Confidence,
    neg_alpha: float = 0.3,
    neg_floor: float = 0.5,
    neg_gate: float = 0.3,
    *,
    rel_threshold: float,
) -> Confidence:
    """Apply negative dampening in a single pass over converged positive_base.

    For pairs above ``neg_gate``, multiplies positive_base by a negative
    factor computed from neighbor mismatches.  The negative factor uses
    ``name_seed`` (fixed name similarity) to check whether neighbors
    match, preventing circular reinforcement.

    Returns the final confidence dict.
    """
    confidence: Confidence = {}
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
        else:
            combined = base
        confidence[(id_a, id_b)] = combined
        confidence[(id_b, id_a)] = combined
    return confidence


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
) -> Confidence:
    """Run similarity propagation on a single unified graph.

    Wraps the positive fixpoint loop and negative dampening in an outer
    epoch loop for progressive merging.  Between epochs, high-confidence
    pairs (above ``merge_threshold``) are committed via union-find,
    enriching neighborhoods for subsequent propagation.

    With default ``merge_threshold=0.9`` and ``max_epochs=5``, pairs that
    don't reach 0.9 never trigger progressive merges, so the epoch loop
    exits after epoch 1 — reproducing prior behavior.

    Returns confidence: (entity_id_a, entity_id_b) -> float in [0, 1].
    Both orderings (a,b) and (b,a) are stored for convenient lookup.
    """
    adjacency = _build_weighted_adjacency(graph, functionality)
    forward_adj = _build_forward_adjacency(graph, functionality)

    # Precompute pairwise relation similarities
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

    # Seed confidence from name similarity.
    name_seed: Confidence = {}
    all_pairs: list[tuple[str, str]] = []
    for i, id_a in enumerate(entity_ids):
        for id_b in entity_ids[i + 1 :]:
            if graph.nodes[id_a].graph_id == graph.nodes[id_b].graph_id:
                continue
            name_sim = max(
                0.0,
                soft_tfidf(graph.nodes[id_a].name, graph.nodes[id_b].name, idf),
            )
            name_seed[(id_a, id_b)] = name_sim
            name_seed[(id_b, id_a)] = name_sim
            all_pairs.append((id_a, id_b))

    uf = UnionFind()

    # Carry-forward confidence from previous epochs (keyed by canonical pairs).
    carried_confidence: Confidence = {}

    for _epoch in range(max_epochs):
        # Build pairs between canonical reps, merging adjacencies.
        canon_pairs: list[tuple[str, str]] = []
        seen_canon: set[tuple[str, str]] = set()
        positive_base: Confidence = {}

        # Precompute best name similarity per canonical pair in a single O(n²) pass.
        best_name_per_canon: dict[tuple[str, str], float] = {}
        for orig_a, orig_b in all_pairs:
            ca, cb = uf.find(orig_a), uf.find(orig_b)
            if ca == cb:
                continue
            cpair = (ca, cb) if ca < cb else (cb, ca)
            best_name_per_canon[cpair] = max(
                best_name_per_canon.get(cpair, 0.0),
                name_seed.get((orig_a, orig_b), 0.0),
            )

        for id_a, id_b in all_pairs:
            ca, cb = uf.find(id_a), uf.find(id_b)
            if ca == cb:
                continue
            pair = (ca, cb) if ca < cb else (cb, ca)
            if pair not in seen_canon:
                seen_canon.add(pair)
                canon_pairs.append(pair)

                best_name = best_name_per_canon.get(pair, 0.0)
                carry = carried_confidence.get(pair, 0.0)
                seed = max(best_name, carry)
                positive_base[pair] = seed
                positive_base[(pair[1], pair[0])] = seed

        if not canon_pairs:
            break

        # Build canonical adjacency: merged entity's neighborhood = union of members'.
        # Deduplicate: for each (canon_node, canon_neighbor, relation), keep the
        # max func_weight. Without deduplication, merging N identical entities
        # creates N duplicate paths that artificially inflate positive evidence.
        canon_adjacency = _build_canonical_adjacency(adjacency, uf)
        canon_forward_adj = _build_canonical_adjacency(forward_adj, uf)

        # Build canonical name_seed for negative evidence
        canon_name_seed: Confidence = {}
        for (a, b), sim in name_seed.items():
            ca, cb = uf.find(a), uf.find(b)
            if ca == cb:
                continue
            key = (ca, cb)
            canon_name_seed[key] = max(canon_name_seed.get(key, 0.0), sim)

        # Positive fixpoint
        propagate_positive(
            canon_pairs,
            positive_base,
            canon_adjacency,
            rel_sim,
            rel_threshold=rel_threshold,
            max_iter=max_iter,
            epsilon=epsilon,
            exp_lambda=exp_lambda,
        )

        # Negative dampening
        confidence = apply_negative(
            canon_pairs,
            positive_base,
            canon_forward_adj,
            rel_sim,
            canon_name_seed,
            neg_alpha=neg_alpha,
            neg_floor=neg_floor,
            neg_gate=neg_gate,
            rel_threshold=rel_threshold,
        )

        # Find new merges above merge_threshold
        new_merges = []
        for ca, cb in canon_pairs:
            if confidence.get((ca, cb), 0.0) >= merge_threshold:
                if uf.find(ca) != uf.find(cb):
                    new_merges.append((ca, cb))

        if not new_merges:
            break

        # Commit merges
        for ca, cb in new_merges:
            uf.union(ca, cb)

        # Carry forward confidence for next epoch
        carried_confidence = {}
        for ca, cb in canon_pairs:
            ra, rb = uf.find(ca), uf.find(cb)
            if ra == rb:
                continue
            key = (ra, rb) if ra < rb else (rb, ra)
            val = confidence.get((ca, cb), 0.0)
            carried_confidence[key] = max(carried_confidence.get(key, 0.0), val)

    # Map final confidence back to all original pairs.
    # For each original pair, look up the canonical pair's confidence.
    final: Confidence = {}
    for id_a, id_b in all_pairs:
        ca, cb = uf.find(id_a), uf.find(id_b)
        if ca == cb:
            # Merged entities — assign confidence 1.0
            final[(id_a, id_b)] = 1.0
            final[(id_b, id_a)] = 1.0
        else:
            score = confidence.get((ca, cb), confidence.get((cb, ca), 0.0))
            final[(id_a, id_b)] = score
            final[(id_b, id_a)] = score

    return final


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
        rel_threshold=rel_cluster_threshold,
        **propagate_kwargs,
    )


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

    match_groups, unified = build_match_groups(graphs, confidence, match_threshold)
    save_graph(unified, output_path, [list(group) for group in match_groups])

    click.echo(f"\n{len(match_groups)} match groups:")
    for members in match_groups:
        names = {unified.nodes[eid].name for eid in members}
        click.echo(f"  {' / '.join(sorted(names))}")

    click.echo(f"\nWrote {output_path}")
