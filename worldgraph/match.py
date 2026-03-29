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
    """An entry in a node's weighted adjacency list.

    ``func_weight`` is used by positive propagation (inverse for outgoing,
    forward for incoming — measures how uniquely the neighbor determines
    this entity).

    ``neg_func_weight`` is used by negative evidence (forward for outgoing,
    inverse for incoming — measures how uniquely this entity determines
    the neighbor, so a missing match on a functional relation penalizes
    heavily).
    """

    entity_id: str
    relation: str
    func_weight: float
    neg_func_weight: float


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
    adj: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    confidence: Confidence,
    alpha: float = 0.3,
    floor: float = 0.5,
    *,
    rel_threshold: float,
    uf: UnionFind,
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
        adj,
        rel_sim,
        confidence,
        alpha,
        floor,
        rel_threshold=rel_threshold,
        uf=uf,
    )
    neg_b = _one_sided_negative(
        id_b,
        id_a,
        adj,
        rel_sim,
        confidence,
        alpha,
        floor,
        rel_threshold=rel_threshold,
        uf=uf,
    )
    return max(neg_a, neg_b)


def _one_sided_negative(
    id_a: str,
    id_b: str,
    adj: dict[str, list[Neighbor]],
    rel_sim: dict[tuple[str, str], float],
    confidence: Confidence,
    alpha: float,
    floor: float,
    *,
    rel_threshold: float,
    uf: UnionFind,
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

    Neighbor entity IDs are resolved through ``uf`` to handle merged
    entities whose adjacency entries may reference pre-merge IDs.
    """
    neighbors_a = adj.get(id_a, [])
    if not neighbors_a:
        return 1.0

    total_weight = 0.0
    weighted_mismatch = 0.0
    for neighbor_a in neighbors_a:
        ra = uf.find(neighbor_a.entity_id)
        if ra == id_a:
            continue
        match_prob = 0.0
        for neighbor_b in adj.get(id_b, []):
            rb = uf.find(neighbor_b.entity_id)
            if rb == id_b:
                continue
            rs = rel_sim.get((neighbor_a.relation, neighbor_b.relation), 0.0)
            if rs < rel_threshold:
                continue
            # Same canonical entity = perfect match (both sides merged).
            nbr_conf = 1.0 if ra == rb else confidence.get((ra, rb), 0.0)
            match_prob += nbr_conf
        match_prob = min(match_prob, 1.0)
        total_weight += neighbor_a.neg_func_weight
        weighted_mismatch += neighbor_a.neg_func_weight * (1.0 - match_prob)

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


def _build_adjacency(
    graph: Graph,
    functionality: dict[str, Functionality],
) -> dict[str, list[Neighbor]]:
    """Build the initial canonical adjacency from graph edges.

    Each edge contributes two entries (one per endpoint).  Duplicate
    entries (same neighbor + same relation) are deduplicated to prevent
    inflated evidence.
    """
    default = Functionality(1.0, 1.0)
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    adjacency: dict[str, list[Neighbor]] = defaultdict(list)
    for edge in graph.edges:
        func = functionality.get(edge.relation, default)
        src, tgt = edge.source, edge.target
        if src == tgt:
            continue
        key_src = (tgt, edge.relation)
        if key_src not in seen[src]:
            seen[src].add(key_src)
            adjacency[src].append(
                Neighbor(tgt, edge.relation, func.inverse, func.forward)
            )
        key_tgt = (src, edge.relation)
        if key_tgt not in seen[tgt]:
            seen[tgt].add(key_tgt)
            adjacency[tgt].append(
                Neighbor(src, edge.relation, func.forward, func.inverse)
            )
    return dict(adjacency)


def _build_pairs(graph: Graph) -> list[tuple[str, str]]:
    """Build cross-graph entity pairs."""
    graph_ids = {eid: node.graph_id for eid, node in graph.nodes.items()}
    entities = sorted(graph.nodes.keys())
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            if graph_ids[a] != graph_ids[b]:
                pairs.append((a, b))
    return pairs


def _seed_confidence(
    graph: Graph,
    idf: dict[str, float],
    pairs: list[tuple[str, str]],
) -> tuple[Confidence, Confidence]:
    """Seed confidence from name similarity.

    Returns (confidence, name_seed) — initially identical.  ``name_seed``
    is kept fixed (modulo remapping on merge) and used by negative evidence
    to prevent circular reinforcement.
    """
    confidence: Confidence = {}
    name_seed: Confidence = {}
    for a, b in pairs:
        best = 0.0
        for na in graph.nodes[a].names:
            for nb in graph.nodes[b].names:
                best = max(best, soft_tfidf(na, nb, idf))
        best = max(0.0, best)
        confidence[(a, b)] = best
        confidence[(b, a)] = best
        name_seed[(a, b)] = best
        name_seed[(b, a)] = best
    return confidence, name_seed


def _remap_confidence(conf: Confidence, uf: UnionFind) -> Confidence:
    """Remap a confidence dict to canonical reps, taking max on collisions."""
    remapped: Confidence = {}
    for (a, b), score in conf.items():
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        old = remapped.get((ra, rb), 0.0)
        remapped[(ra, rb)] = max(old, score)
        remapped[(rb, ra)] = max(old, score)
    return remapped


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
) -> tuple[Confidence, UnionFind]:
    """Run similarity propagation with inline progressive merging.

    A single propagation loop alternates positive evidence accumulation
    and negative dampening.  When pairs exceed ``merge_threshold``, they
    are merged via union-find and the canonical adjacency is updated
    incrementally — no epoch rebuild required.

    Positive updates use the monotone max rule (new = max(structural, old))
    so positive evidence never decreases.  Negative dampening is applied
    as a multiplicative factor after each positive pass.

    On merge, the canonical adjacency for the new representative is built
    by combining and deduplicating the adjacency lists of the merged
    entities — O(degree) per merge, not O(|edges|).

    Returns (confidence, union_find) where confidence maps original
    entity-ID pairs to scores and union_find tracks all committed merges.
    """
    rel_sim = _build_rel_sim(graph, relation_embeddings)
    uf = UnionFind()
    for eid in graph.nodes:
        uf.find(eid)

    canonical_adj = _build_adjacency(graph, functionality)
    pairs = _build_pairs(graph)

    if not pairs:
        return {}, uf

    confidence, name_seed = _seed_confidence(graph, idf, pairs)

    for _ in range(max_iter):
        prev = dict(confidence)
        pos_changed = False

        # --- Positive propagation ---
        for ca, cb in pairs:
            strength_sum = 0.0
            for nbr_a in canonical_adj.get(ca, []):
                ra = uf.find(nbr_a.entity_id)
                if ra == ca:
                    continue
                for nbr_b in canonical_adj.get(cb, []):
                    rb = uf.find(nbr_b.entity_id)
                    if rb == cb:
                        continue
                    rs = rel_sim.get((nbr_a.relation, nbr_b.relation), 0.0)
                    if rs < rel_threshold:
                        continue
                    nc = prev.get((ra, rb), 0.0)
                    if nc <= 0.0:
                        continue
                    weight = min(nbr_a.func_weight, nbr_b.func_weight)
                    strength_sum += weight * nc

            positive = (
                1.0 - math.exp(-exp_lambda * strength_sum) if strength_sum > 0 else 0.0
            )
            old = prev[(ca, cb)]
            new_val = max(positive, old)
            confidence[(ca, cb)] = new_val
            confidence[(cb, ca)] = new_val
            if abs(new_val - old) > epsilon:
                pos_changed = True

        # Keep iterating positive until convergence before applying
        # negative dampening and checking for merges.
        if pos_changed:
            continue

        # --- Negative dampening (positive has converged) ---
        for ca, cb in pairs:
            base = confidence[(ca, cb)]
            if base > neg_gate:
                neg = compute_negative_factor(
                    ca,
                    cb,
                    canonical_adj,
                    rel_sim,
                    name_seed,
                    alpha=neg_alpha,
                    floor=neg_floor,
                    rel_threshold=rel_threshold,
                    uf=uf,
                )
                combined = base * neg
                confidence[(ca, cb)] = combined
                confidence[(cb, ca)] = combined

        # --- Progressive merging ---
        new_merges = [
            (ca, cb)
            for ca, cb in pairs
            if confidence.get((ca, cb), 0.0) >= merge_threshold
            and uf.find(ca) != uf.find(cb)
        ]

        if new_merges:
            all_merged = {e for ca, cb in new_merges for e in (ca, cb)}
            for ca, cb in new_merges:
                uf.union(ca, cb)

            # Update canonical_adj incrementally: combine + dedup.
            merge_groups: dict[str, list[str]] = defaultdict(list)
            for e in all_merged:
                merge_groups[uf.find(e)].append(e)
            for new_canon, old_canons in merge_groups.items():
                combined: list[Neighbor] = []
                for oc in old_canons:
                    combined.extend(canonical_adj.get(oc, []))
                seen: set[tuple[str, str]] = set()
                deduped: list[Neighbor] = []
                for nbr in combined:
                    canon_nbr = uf.find(nbr.entity_id)
                    if canon_nbr == new_canon:
                        continue
                    key = (canon_nbr, nbr.relation)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(
                            Neighbor(
                                canon_nbr,
                                nbr.relation,
                                nbr.func_weight,
                                nbr.neg_func_weight,
                            )
                        )
                canonical_adj[new_canon] = deduped

            # Remap pairs, confidence, name_seed to canonical reps.
            pair_set: set[tuple[str, str]] = set()
            new_pairs: list[tuple[str, str]] = []
            for a, b in pairs:
                ra, rb = uf.find(a), uf.find(b)
                if ra == rb:
                    continue
                pair = (min(ra, rb), max(ra, rb))
                if pair not in pair_set:
                    pair_set.add(pair)
                    new_pairs.append(pair)
            pairs = new_pairs

            confidence = _remap_confidence(confidence, uf)
            name_seed = _remap_confidence(name_seed, uf)

            # Re-seed: confidence should never fall below name similarity.
            # Without this, negative dampening compounds across merge cycles.
            for ca, cb in pairs:
                ns = name_seed.get((ca, cb), 0.0)
                if ns > confidence.get((ca, cb), 0.0):
                    confidence[(ca, cb)] = ns
                    confidence[(cb, ca)] = ns

            if not pairs:
                break
            continue

        # Positive converged, no new merges — done.
        break

    # Expand canonical-rep confidence to original entity-ID pairs.
    members: dict[str, list[str]] = defaultdict(list)
    for eid in graph.nodes:
        members[uf.find(eid)].append(eid)

    final: Confidence = {}
    for (ca, cb), score in confidence.items():
        if ca == cb:
            continue
        for ma in members.get(ca, [ca]):
            for mb in members.get(cb, [cb]):
                if graph.nodes[ma].graph_id == graph.nodes[mb].graph_id:
                    continue
                old = final.get((ma, mb), 0.0)
                new_val = max(old, score)
                final[(ma, mb)] = new_val
                final[(mb, ma)] = new_val

    # Merged pairs get 1.0 — they were committed with high confidence.
    for group_members in members.values():
        if len(group_members) < 2:
            continue
        for i, ma in enumerate(group_members):
            for mb in group_members[i + 1 :]:
                if graph.nodes[ma].graph_id == graph.nodes[mb].graph_id:
                    continue
                final[(ma, mb)] = 1.0
                final[(mb, ma)] = 1.0

    return final, uf


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
    )

    match_groups, unified = build_match_groups(graphs, confidence, match_threshold)
    save_graph(unified, output_path, [list(group) for group in match_groups])

    click.echo(f"\n{len(match_groups)} match groups:")
    for members in match_groups:
        names = {n for eid in members for n in unified.nodes[eid].names}
        click.echo(f"  {' / '.join(sorted(names))}")

    click.echo(f"\nWrote {output_path}")
