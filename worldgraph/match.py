"""Stage 2: Entity alignment via damped similarity propagation.

A single confidence score per entity pair is iteratively refined using
damped fixed-point iteration.  Each step computes structural evidence
from matching neighbor pairs (positive: neighbors likely match, weighted
by inverse functionality; negative: neighbors likely don't match, weighted
by forward functionality) and blends it with the previous score via a
damping factor.

Name similarity seeds the initial scores and serves as the baseline that
structural evidence modulates up or down.

Relation similarity is treated as binary via a single threshold that defines
equivalence classes over free-text relation phrases.  This threshold is used
consistently for functionality pooling and propagation gating.
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

    ``pos_weight`` weights positive evidence — inverse functionality for
    outgoing edges, forward functionality for incoming edges.

    ``neg_weight`` weights negative evidence — forward functionality for
    outgoing edges, inverse functionality for incoming edges.
    """

    entity_id: str
    relation: str
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
                Neighbor(
                    tgt, edge.relation, pos_weight=func.inverse, neg_weight=func.forward
                )
            )
        key_tgt = (src, edge.relation)
        if key_tgt not in seen[tgt]:
            seen[tgt].add(key_tgt)
            adjacency[tgt].append(
                Neighbor(
                    src, edge.relation, pos_weight=func.forward, neg_weight=func.inverse
                )
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
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    rel_threshold: float = 0.8,
    max_iter: int = 30,
    epsilon: float = 1e-4,
    exp_lambda: float = 1.0,
    merge_threshold: float = 0.9,
    damping: float = 0.5,
) -> tuple[Confidence, UnionFind]:
    """Run damped similarity propagation with progressive merging.

    A single confidence score per entity pair integrates both positive and
    negative structural evidence.  Each iteration computes a new score from
    neighbor confidences and blends it with the old score via damping::

        computed = seed + pos_agg * (1 - seed) - neg_agg * seed
        new = (1 - damping) * old + damping * computed

    Name similarity (``seed``) is the baseline: positive evidence pushes
    toward 1.0, negative evidence pushes toward 0.0.  With no structural
    evidence the fixpoint equals the seed.

    On merge, the canonical adjacency for the new representative is built
    by combining and deduplicating the adjacency lists of the merged
    entities — O(degree) per merge, not O(|edges|).

    Returns (confidence, union_find) where confidence maps original
    entity-ID pairs to scores and union_find tracks all merges.
    """
    rel_sim = _build_rel_sim(graph, relation_embeddings)
    uf = UnionFind()
    for eid in graph.nodes:
        uf.find(eid)

    canonical_adj = _build_adjacency(graph, functionality)
    pairs = _build_pairs(graph)

    if not pairs:
        return {}, uf

    conf, name_sim = _seed_confidence(graph, idf, pairs)

    for _ in range(max_iter):
        prev = dict(conf)
        changed = False

        for ca, cb in pairs:
            pos_strength = 0.0
            neg_strength = 0.0

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

                    if ra == rb:
                        # Neighbors already merged — strongest positive evidence.
                        pos_strength += min(nbr_a.pos_weight, nbr_b.pos_weight)
                        continue

                    nc = prev.get((ra, rb), 0.0)
                    if nc > 0.5:
                        pos_strength += min(nbr_a.pos_weight, nbr_b.pos_weight) * nc

                    neg_nc = 1.0 - nc
                    if neg_nc > 0.5:
                        neg_strength += min(nbr_a.neg_weight, nbr_b.neg_weight) * neg_nc

            # Exp-sum aggregation + seed-as-baseline combination.
            pos_agg = (
                1.0 - math.exp(-exp_lambda * pos_strength) if pos_strength > 0 else 0.0
            )
            neg_agg = (
                1.0 - math.exp(-exp_lambda * neg_strength) if neg_strength > 0 else 0.0
            )

            seed = name_sim[(ca, cb)]
            computed = seed + pos_agg * (1.0 - seed) - neg_agg * seed
            computed = max(0.0, min(1.0, computed))

            old = prev[(ca, cb)]
            new_val = (1.0 - damping) * old + damping * computed
            conf[(ca, cb)] = new_val
            conf[(cb, ca)] = new_val

            if abs(new_val - old) > epsilon:
                changed = True

        if changed:
            continue

        # --- Progressive merging (directly on single score) ---
        new_merges = [
            (ca, cb)
            for ca, cb in pairs
            if conf[(ca, cb)] >= merge_threshold and uf.find(ca) != uf.find(cb)
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
                                nbr.pos_weight,
                                nbr.neg_weight,
                            )
                        )
                canonical_adj[new_canon] = deduped

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
    pooling and propagation gating.

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
