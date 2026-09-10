"""Stage 2: Entity alignment via damped similarity propagation.

A single confidence score per entity pair is iteratively refined using
damped fixed-point iteration.  Each step computes per-neighbor structural
evidence (positive: neighbor's best counterpart is a likely match;
negative: neighbor has no good counterpart) and blends it with the
previous score via a damping factor.

Name similarity seeds the initial scores; structural evidence is required
for merging.

Relation similarity is treated as binary via a single threshold that defines
equivalence classes over free-text relation phrases.  This threshold is used
consistently for relation clustering, functionality pooling, adjacency
deduplication, and propagation gating.
"""

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
from dotenv import load_dotenv

from worldgraph.constants import (
    MERGE_THRESHOLD,
    RELATION_TEMPLATE,
    RELATION_THRESHOLD,
)
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

    ``pos_weight`` weights positive evidence — inverse functionality for
    outgoing edges, forward functionality for incoming edges.

    ``neg_weight`` weights negative evidence — forward functionality for
    outgoing edges, inverse functionality for incoming edges.
    """

    entity_id: str
    relation: str
    temporal: str
    pos_weight: float
    neg_weight: float


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
    rel_clusters: dict[str, int],
) -> dict[str, Functionality]:
    """Compute functionality and inverse functionality for each relation phrase.

    Functionality ≈ 1 / avg_out_degree: for a given source name, how many
    distinct target names does it map to via this relation pool? High means
    the relation uniquely determines the target — strong forward evidence.

    Inverse functionality ≈ 1 / avg_in_degree: for a given target name, how
    many distinct source names map to it via this relation pool? High means
    the relation uniquely determines the source — strong backward evidence.

    Entity names (not IDs) are used so that the same entity mentioned across
    multiple graphs pools its statistics.  Edges whose relation phrases belong
    to the same cluster and temporal class are pooled together.

    Returns dict from (phrase, temporal) to Functionality(forward, inverse).
    """
    # Collect all (source_name, target_name) pairs per (relation cluster, temporal).
    pool_pairs: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    for graph in graphs:
        for edge in graph.edges.values():
            cid = rel_clusters.get(edge.relation, -1)
            source_name = graph.nodes[edge.source].names[0]
            target_name = graph.nodes[edge.target].names[0]
            pool_pairs[(cid, edge.temporal)].append((source_name, target_name))

    # Compute functionality per (cluster, temporal) pool, then map back to
    # each (phrase, temporal).
    pool_func: dict[tuple[int, str], Functionality] = {}
    for pool, pairs in pool_pairs.items():
        targets_per_source: dict[str, set[str]] = defaultdict(set)
        sources_per_target: dict[str, set[str]] = defaultdict(set)
        for source_name, target_name in pairs:
            targets_per_source[source_name].add(target_name)
            sources_per_target[target_name].add(source_name)
        avg_out_degree = sum(
            len(targets) for targets in targets_per_source.values()
        ) / len(targets_per_source)
        avg_in_degree = sum(
            len(sources) for sources in sources_per_target.values()
        ) / len(sources_per_target)
        pool_func[pool] = Functionality(1.0 / avg_out_degree, 1.0 / avg_in_degree)

    observed = {
        (edge.relation, edge.temporal)
        for graph in graphs
        for edge in graph.edges.values()
    }
    return {
        (rel, temporal): pool_func[
            (rel_clusters.get(rel, -1), temporal)
        ]
        for (rel, temporal) in observed
    }


# ---------------------------------------------------------------------------
# Similarity propagation
# ---------------------------------------------------------------------------


def build_unified_graph(graphs: list[Graph]) -> Graph:
    """Combine N article graphs into one. Node IDs are UUIDs — unique across graphs.

    The unified graph gets the fixed id "unified" so that identical inputs
    produce identical pipeline output.
    """
    unified = Graph(id="unified")
    for graph in graphs:
        unified.nodes.update(graph.nodes)
        unified.edges.update(graph.edges)
    return unified


def build_rel_sim(
    relations: set[str],
    relation_embeddings: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Precompute pairwise relation similarities for a set of relation phrases."""
    rel_sim: dict[tuple[str, str], float] = {}
    for rel_a in relations:
        embedding_a = relation_embeddings.get(rel_a)
        if embedding_a is None:
            continue
        for rel_b in relations:
            embedding_b = relation_embeddings.get(rel_b)
            if embedding_b is None:
                continue
            rel_sim[(rel_a, rel_b)] = max(0.0, float(np.dot(embedding_a, embedding_b)))
    return rel_sim


def build_rel_clusters(
    rel_sim: dict[tuple[str, str], float],
    rel_threshold: float,
) -> dict[str, int]:
    """Assign each relation phrase to an equivalence class.

    Greedy single-linkage: each phrase joins the first cluster whose
    representative has similarity >= threshold.  Returns a mapping from
    phrase to integer cluster ID.
    """
    clusters: list[str] = []  # representative phrase per cluster
    mapping: dict[str, int] = {}
    for rel in sorted({r for pair in rel_sim for r in pair}):
        for i, rep in enumerate(clusters):
            if rel_sim.get((rel, rep), 0.0) >= rel_threshold:
                mapping[rel] = i
                break
        else:
            mapping[rel] = len(clusters)
            clusters.append(rel)
    return mapping


def _dedup_neighbors(
    neighbors: list[Neighbor],
    rel_clusters: dict[str, int],
) -> list[Neighbor]:
    """Deduplicate neighbor entries by (neighbor_id, relation cluster, temporal).

    Entries to the same neighbor via equivalent relations in the same
    temporal class represent the same structural evidence.  Keeps the
    max-weight entry per (neighbor, cluster, temporal) group.
    """
    best: dict[tuple[str, int, str], Neighbor] = {}
    for nbr in neighbors:
        key = (nbr.entity_id, rel_clusters.get(nbr.relation, -1), nbr.temporal)
        prev = best.get(key)
        if prev is None or nbr.pos_weight > prev.pos_weight:
            best[key] = nbr
    return list(best.values())


def _build_adjacency(
    graph: Graph,
    functionality: dict[tuple[str, str], Functionality],
    rel_clusters: dict[str, int],
) -> dict[str, list[Neighbor]]:
    """Build the initial canonical adjacency from graph edges.

    Each edge contributes two entries (one per endpoint).  Entries to the
    same neighbor via equivalent relations in the same temporal class are
    deduplicated by (relation cluster, temporal) to prevent inflated evidence.
    """
    default = Functionality(1.0, 1.0)
    adjacency: dict[str, list[Neighbor]] = defaultdict(list)
    for edge in graph.edges.values():
        func = functionality.get((edge.relation, edge.temporal), default)
        src, tgt = edge.source, edge.target
        if src == tgt:
            continue
        adjacency[src].append(
            Neighbor(
                tgt,
                edge.relation,
                edge.temporal,
                pos_weight=func.inverse,
                neg_weight=func.forward,
            )
        )
        adjacency[tgt].append(
            Neighbor(
                src,
                edge.relation,
                edge.temporal,
                pos_weight=func.forward,
                neg_weight=func.inverse,
            )
        )
    return {
        eid: _dedup_neighbors(nbrs, rel_clusters) for eid, nbrs in adjacency.items()
    }


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

    Returns (conf, name_sim) where both are initialized from the best
    soft-TF-IDF score across all name pairs.  ``name_sim`` is kept as a
    read-only baseline for the seed-as-baseline update formula.
    """
    conf: Confidence = {}
    name_sim: Confidence = {}
    for a, b in pairs:
        best = 0.0
        for na in graph.nodes[a].names:
            for nb in graph.nodes[b].names:
                best = max(best, soft_tfidf(na, nb, idf))
        best = max(0.0, best)
        conf[(a, b)] = best
        conf[(b, a)] = best
        name_sim[(a, b)] = best
        name_sim[(b, a)] = best
    return conf, name_sim


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
    rel_clusters: dict[str, int],
    functionality: dict[tuple[str, str], Functionality],
    max_iter: int = 30,
    epsilon: float = 1e-4,
    exp_lambda: float = 1.0,
    merge_threshold: float = MERGE_THRESHOLD,
    damping: float = 0.5,
    prior_strength: float = 1.0,
) -> tuple[Confidence, list[MatchGroup]]:
    """Run damped similarity propagation with progressive merging.

    A single confidence score per entity pair integrates both positive and
    negative structural evidence.  For each neighbor of entity A, we find
    its best counterpart among B's neighbors (same relation cluster) and
    contribute once — positive if the best counterpart confidence exceeds
    0.5, negative if it falls below.  Neighbors that resolve to either
    entity in the pair are excluded to prevent circular self-reference.

    Evidence is computed **bidirectionally** (A→B and B→A) and averaged.
    Positive evidence is weighted by **Bayesian shrinkage**
    ``n / (n + κ)`` where ``n`` is the number of neighbors that found a
    same-cluster counterpart and ``κ`` (``prior_strength``) controls how
    many tested neighbors are needed before trusting structural matches.
    Negative evidence has full weight — a mismatch on a functional
    relation is decisive and does not need corroboration.

    Each directional score is::

        weight = n_tested / (n_tested + prior_strength)
        evidence = pos_agg * (1 - seed) * weight - neg_agg * seed
        computed_dir = seed + evidence

    The final value is the simple average of both directions, blended
    with the previous score via damping::

        computed = (computed_fwd + computed_bwd) / 2
        new = (1 - damping) * old + damping * computed

    Name similarity (``seed``) initialises the confidence scores so
    propagation has signal to start with, and anchors the per-iteration
    formula.  But merging requires structural evidence: pairs with zero
    tested neighbors are never merged, regardless of name similarity.

    On merge, the canonical adjacency for the new representative is built
    by combining and deduplicating the adjacency lists of the merged
    entities — O(degree) per merge, not O(|edges|).

    Returns (confidence, match_groups). The union-find maintained during
    propagation is the **sole merge authority**: a pair is merged iff it
    crossed ``merge_threshold`` with at least one tested neighbor, and
    ``match_groups`` lists exactly those committed groups (size > 1).
    There is no second, post-hoc grouping pass.
    """
    uf = UnionFind()
    for eid in graph.nodes:
        uf.find(eid)

    canonical_adj = _build_adjacency(graph, functionality, rel_clusters)
    pairs = _build_pairs(graph)

    if not pairs:
        return {}, []

    conf, name_sim = _seed_confidence(graph, idf, pairs)

    def _directional_evidence(
        src: str,
        tgt: str,
        prev: Confidence,
    ) -> tuple[float, float, int]:
        """Per-neighbor best-counterpart evidence from src's perspective.

        For each neighbor of *src*, find its best counterpart among *tgt*'s
        neighbors in the same relation cluster and contribute positive
        (nc > 0.5) or negative (nc < 0.5) evidence once.

        Returns (pos_strength, neg_strength, n_with_counterpart).
        """
        pos_strength = 0.0
        neg_strength = 0.0
        n_with_counterpart = 0
        nbrs_tgt = canonical_adj.get(tgt, [])
        for nbr_s in canonical_adj.get(src, []):
            rs = uf.find(nbr_s.entity_id)
            if rs == src or rs == tgt:
                continue

            cluster_s = rel_clusters.get(nbr_s.relation, -1)
            best_nc: float | None = None
            best_pos_w = 0.0
            best_neg_w = 0.0
            for nbr_t in nbrs_tgt:
                if rel_clusters.get(nbr_t.relation, -2) != cluster_s:
                    continue
                if nbr_t.temporal != nbr_s.temporal:
                    continue
                rt = uf.find(nbr_t.entity_id)
                if rt == tgt or rt == src:
                    continue
                nc = 1.0 if rs == rt else prev.get((rs, rt), 0.0)
                if best_nc is None or nc > best_nc:
                    best_nc = nc
                    best_pos_w = min(nbr_s.pos_weight, nbr_t.pos_weight)
                    best_neg_w = min(nbr_s.neg_weight, nbr_t.neg_weight)

            if best_nc is None:
                continue

            n_with_counterpart += 1
            if best_nc > 0.5:
                pos_strength += best_pos_w * best_nc
            else:
                neg_nc = 1.0 - best_nc
                if neg_nc > 0.5:
                    neg_strength += best_neg_w * neg_nc

        return pos_strength, neg_strength, n_with_counterpart

    n_tested: dict[tuple[str, str], int] = {}

    for _ in range(max_iter):
        prev = dict(conf)
        changed = False

        for ca, cb in pairs:
            pos_fwd, neg_fwd, n_cp_fwd = _directional_evidence(ca, cb, prev)
            pos_bwd, neg_bwd, n_cp_bwd = _directional_evidence(cb, ca, prev)

            n_tested[(ca, cb)] = n_cp_fwd + n_cp_bwd
            seed = name_sim[(ca, cb)]

            # Bidirectional: shrinkage-weighted average of both perspectives.
            # Shrinkage applies only to positive evidence (structural
            # matches need corroboration); negative evidence has full
            # weight (a mismatch on a functional relation is decisive).
            dir_sum = 0.0
            for pos_s, neg_s, n_cp in (
                (pos_fwd, neg_fwd, n_cp_fwd),
                (pos_bwd, neg_bwd, n_cp_bwd),
            ):
                w = n_cp / (n_cp + prior_strength) if n_cp > 0 else 0.0
                pa = 1.0 - math.exp(-exp_lambda * pos_s) if pos_s > 0 else 0.0
                na = 1.0 - math.exp(-exp_lambda * neg_s) if neg_s > 0 else 0.0
                evidence = pa * (1.0 - seed) * w - na * seed
                dir_sum += max(0.0, min(1.0, seed + evidence))
            computed = dir_sum / 2

            old = prev[(ca, cb)]
            new_val = (1.0 - damping) * old + damping * computed
            conf[(ca, cb)] = new_val
            conf[(cb, ca)] = new_val

            if abs(new_val - old) > epsilon:
                changed = True

        if changed:
            continue

        # --- Progressive merging (requires structural evidence) ---
        new_merges = [
            (ca, cb)
            for ca, cb in pairs
            if conf[(ca, cb)] >= merge_threshold
            and uf.find(ca) != uf.find(cb)
            and n_tested.get((ca, cb), 0) > 0
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
                remapped = [
                    Neighbor(
                        uf.find(nbr.entity_id),
                        nbr.relation,
                        nbr.temporal,
                        nbr.pos_weight,
                        nbr.neg_weight,
                    )
                    for nbr in combined
                    if uf.find(nbr.entity_id) != new_canon
                ]
                canonical_adj[new_canon] = _dedup_neighbors(remapped, rel_clusters)

            # Remap pairs and confidence dicts to canonical reps.
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

            conf = _remap_confidence(conf, uf)
            name_sim = _remap_confidence(name_sim, uf)

            if not pairs:
                break
            continue

        # Converged, no new merges — done.
        break

    # Expand canonical-rep confidence to original entity-ID pairs.
    members: dict[str, list[str]] = defaultdict(list)
    for eid in graph.nodes:
        members[uf.find(eid)].append(eid)

    final: Confidence = {}
    for (ca, cb), score in conf.items():
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

    match_groups = [set(m) for m in members.values() if len(m) > 1]
    return final, match_groups


# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------


def match_graphs(
    graphs: list[Graph],
    embedder: Embedder,
    rel_cluster_threshold: float = RELATION_THRESHOLD,
    **propagate_kwargs,
) -> tuple[Confidence, list[MatchGroup], Graph]:
    """Core matching pipeline: graphs → (confidence, match groups, unified graph).

    Builds unified graph, computes IDF / relation embeddings / functionality /
    relation clusters, and runs similarity propagation.
    ``rel_cluster_threshold`` is the single relation equivalence threshold:
    relation pairs with embedding similarity above this value are assigned
    to the same cluster for functionality pooling, adjacency deduplication,
    and propagation gating.

    ``match_groups`` comes straight from the union-find inside propagation —
    the sole merge authority (one threshold, gated on structural evidence).
    """
    unified = build_unified_graph(graphs)

    all_names = [
        name for graph in graphs for node in graph.nodes.values() for name in node.names
    ]
    all_relations = sorted(
        {edge.relation for graph in graphs for edge in graph.edges.values()}
    )

    idf = build_idf(all_names)
    relation_embeddings = embedder.embed(all_relations, template=RELATION_TEMPLATE)
    rel_sim = build_rel_sim(set(all_relations), relation_embeddings)
    rel_clusters = build_rel_clusters(rel_sim, rel_cluster_threshold)
    functionality = compute_functionality(graphs, rel_clusters)

    confidence, match_groups = propagate_similarity(
        unified,
        idf,
        rel_clusters,
        functionality,
        **propagate_kwargs,
    )
    return confidence, match_groups, unified


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_matching(
    graph_files: list[Path],
    output_path: Path,
    relation_threshold: float,
    merge_threshold: float,
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

    _confidence, match_groups, unified = match_graphs(
        graphs,
        embedder,
        rel_cluster_threshold=relation_threshold,
        max_iter=max_iter,
        epsilon=epsilon,
        merge_threshold=merge_threshold,
    )
    save_graph(unified, output_path, [list(group) for group in match_groups])

    click.echo(f"\n{len(match_groups)} match groups:")
    for members in match_groups:
        names = {n for eid in members for n in unified.nodes[eid].names}
        click.echo(f"  {' / '.join(sorted(names))}")

    click.echo(f"\nWrote {output_path}")
