"""Stage 2: Entity alignment via PARIS-style similarity propagation.

Entity names are represented as literal nodes connected by "is named"
edges, so name similarity flows through the same graph structure as
everything else.  Propagate structural evidence via exponential sum,
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
from worldgraph.graph import Graph, LiteralNode, entity_names, load_graphs, save_graph

load_dotenv()


class Functionality(NamedTuple):
    forward: float
    inverse: float


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
    similar: dict[str, list[str]] = {r: [] for r in all_relations}
    for i, r in enumerate(all_relations):
        for j, r2 in enumerate(all_relations):
            if (
                float(np.dot(relation_embeddings[r], relation_embeddings[r2]))
                >= threshold
            ):
                similar[r].append(r2)

    # Collect all (src_name, tgt_name) pairs per relation phrase
    phrase_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for g in graphs:
        for edge in g.edges:
            for src_name in entity_names(g, edge.source):
                for tgt_name in entity_names(g, edge.target):
                    phrase_pairs[edge.relation].append((src_name, tgt_name))

    result: dict[str, Functionality] = {}
    for r in all_relations:
        pooled = [pair for r2 in similar[r] for pair in phrase_pairs.get(r2, [])]
        if not pooled:
            result[r] = Functionality(1.0, 1.0)
            continue
        targets_per_source: dict[str, set[str]] = defaultdict(set)
        sources_per_target: dict[str, set[str]] = defaultdict(set)
        for src, tgt in pooled:
            targets_per_source[src].add(tgt)
            sources_per_target[tgt].add(src)
        avg_out_degree = sum(len(v) for v in targets_per_source.values()) / len(
            targets_per_source
        )
        avg_in_degree = sum(len(v) for v in sources_per_target.values()) / len(
            sources_per_target
        )
        result[r] = Functionality(1.0 / avg_out_degree, 1.0 / avg_in_degree)

    return result


# ---------------------------------------------------------------------------
# Similarity propagation
# ---------------------------------------------------------------------------


def _build_adjacency(
    graph: Graph,
    functionality: dict[str, Functionality],
) -> dict[str, list[tuple[str, str, float]]]:
    """Build per-entity adjacency list with direction-appropriate functionality.

    PARIS semantics: the functionality weight measures "if my neighbor matches,
    how strong is the evidence that I match?"

    For edge src --r--> tgt:
      - src uses inverse functionality: "given the target, how unique is the
        source?"  If fun⁻¹(r) ≈ 1, a target match strongly implies a source match.
      - tgt uses forward functionality: "given the source, how unique is the
        target?"  If fun(r) ≈ 1, a source match strongly implies a target match.

    Returns {entity_id: [(neighbor_id, relation, func_weight), ...]}.
    """
    default = Functionality(1.0, 1.0)
    adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for edge in graph.edges:
        func = functionality.get(edge.relation, default)
        adj[edge.source].append((edge.target, edge.relation, func.inverse))
        adj[edge.target].append((edge.source, edge.relation, func.forward))
    return adj


def build_unified_graph(graphs: list[Graph]) -> Graph:
    """Combine N article graphs into one. Node IDs are UUIDs — unique across graphs."""
    unified = Graph()
    for g in graphs:
        unified.nodes.update(g.nodes)
        unified.edges.extend(g.edges)
    return unified


def propagate(
    graph: Graph,
    literal_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    max_iter: int = 30,
    epsilon: float = 1e-4,
    rel_gate: float = 0.8,
    confidence_gate: float = 0.5,
    exp_lambda: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Run similarity propagation on a single unified graph.

    Compares entity pairs from different source graphs (based on
    node.graph_id).  Literal-literal confidence is computed on demand
    from embeddings.  Entity-entity confidence is updated iteratively.

    Returns confidence: (entity_id_a, entity_id_b) -> float in [0, 1].
    Both orderings (a,b) and (b,a) are stored for convenient lookup.
    """
    adj = _build_adjacency(graph, functionality)

    # Precompute which relation pairs pass the gate
    all_rels = {edge.relation for edge in graph.edges}
    rel_passes_gate: set[tuple[str, str]] = set()
    for ra in all_rels:
        emb_a = relation_embeddings.get(ra)
        if emb_a is None:
            continue
        for rb in all_rels:
            emb_b = relation_embeddings.get(rb)
            if emb_b is None:
                continue
            if float(np.dot(emb_a, emb_b)) >= rel_gate:
                rel_passes_gate.add((ra, rb))

    # Identify entity nodes (non-literal)
    entity_ids = [
        nid for nid, n in graph.nodes.items() if not isinstance(n, LiteralNode)
    ]

    # Sparse confidence dict: only cross-graph entity-entity pairs.
    # Both orderings stored for convenient neighbor lookup.
    confidence: dict[tuple[str, str], float] = {}
    pairs: list[tuple[str, str]] = []
    for i, id_a in enumerate(entity_ids):
        for id_b in entity_ids[i + 1 :]:
            if graph.nodes[id_a].graph_id == graph.nodes[id_b].graph_id:
                continue
            confidence[(id_a, id_b)] = 0.0
            confidence[(id_b, id_a)] = 0.0
            pairs.append((id_a, id_b))

    def _get_confidence(a: str, b: str) -> float:
        """Look up confidence for any node pair, computing literal sim on demand."""
        if (a, b) in confidence:
            return confidence[(a, b)]
        node_a = graph.nodes[a]
        node_b = graph.nodes[b]
        if isinstance(node_a, LiteralNode) and isinstance(node_b, LiteralNode):
            emb_a = literal_embeddings.get(node_a.label)
            emb_b = literal_embeddings.get(node_b.label)
            if emb_a is not None and emb_b is not None:
                return max(0.0, float(np.dot(emb_a, emb_b)))
        return 0.0

    for _ in range(max_iter):
        changed = False

        for id_a, id_b in pairs:
            strength_sum = 0.0

            for nbr_a, rel_a, func_a in adj.get(id_a, []):
                for nbr_b, rel_b, func_b in adj.get(id_b, []):
                    if (rel_a, rel_b) not in rel_passes_gate:
                        continue
                    nbr_conf = _get_confidence(nbr_a, nbr_b)
                    if nbr_conf < confidence_gate:
                        continue
                    func_w = min(func_a, func_b)
                    strength_sum += func_w * nbr_conf

            combined = (
                1.0 - math.exp(-exp_lambda * strength_sum) if strength_sum > 0 else 0.0
            )

            old = confidence[(id_a, id_b)]
            if combined > old + epsilon:
                confidence[(id_a, id_b)] = combined
                confidence[(id_b, id_a)] = combined
                changed = True

        if not changed:
            break

    return confidence


# ---------------------------------------------------------------------------
# Top-level matching pipeline
# ---------------------------------------------------------------------------


def run_matching(
    graphs_dir: Path,
    output_path: Path,
    threshold: float,
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> None:
    """Load graphs, build unified graph, run single-pass matching, save."""
    graphs = load_graphs(graphs_dir)
    n_initial = len(graphs)
    click.echo(f"Loaded {n_initial} graphs from {graphs_dir}/")
    for g in graphs:
        entities = [n for n in g.nodes.values() if not isinstance(n, LiteralNode)]
        click.echo(f"  {g.id}: {len(entities)} entities, {len(g.edges)} edges")

    unified = build_unified_graph(graphs)

    embedder = Embedder(os.environ["EMBEDDING_MODEL"])

    all_literals = sorted(
        {
            n.label
            for g in graphs
            for n in g.nodes.values()
            if isinstance(n, LiteralNode)
        }
    )
    all_relations = sorted({edge.relation for g in graphs for edge in g.edges})

    literal_embeddings = embedder.embed(all_literals)
    relation_embeddings = embedder.embed(all_relations, template=RELATION_TEMPLATE)
    functionality = compute_functionality(graphs, relation_embeddings, threshold)

    confidence = propagate(
        unified,
        literal_embeddings,
        relation_embeddings,
        functionality,
        max_iter=max_iter,
        epsilon=epsilon,
        rel_gate=threshold,
    )

    # Select matches and build union-find
    uf = UnionFind()
    for (id_a, id_b), score in confidence.items():
        if score >= threshold:
            uf.union(id_a, id_b)

    # Group matched entities
    entity_ids = [
        nid for nid, n in unified.nodes.items() if not isinstance(n, LiteralNode)
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for eid in entity_ids:
        groups[uf.find(eid)].append(eid)
    match_groups = [members for members in groups.values() if len(members) > 1]

    save_graph(unified, output_path, match_groups)

    click.echo(f"\n{len(match_groups)} match groups:")
    for members in match_groups:
        names = []
        for eid in members:
            names.extend(entity_names(unified, eid))
        click.echo(f"  {' / '.join(sorted(set(names)))}")

    click.echo(f"\nWrote {output_path}")
