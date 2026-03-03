"""Stage 2: Entity alignment via PARIS-style similarity propagation.

Entity names are represented as literal nodes connected by "is named"
edges, so name similarity flows through the same graph structure as
everything else.  Propagate structural evidence via exponential sum,
threshold, merge via union-find.
"""

import json
import math
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

load_dotenv()


class Functionality(NamedTuple):
    forward: float
    inverse: float


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: str
    graph_id: str


@dataclass
class LiteralNode(Node):
    label: str = ""


@dataclass
class Edge:
    source: str  # node id
    target: str  # node id
    relation: str


@dataclass
class Graph:
    id: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_entity(self, name: str) -> Node:
        """Add an entity node with an "is named" literal edge."""
        entity = Node(id=str(uuid.uuid4()), graph_id=self.id)
        lit = LiteralNode(id=str(uuid.uuid4()), graph_id=self.id, label=name)
        self.nodes[entity.id] = entity
        self.nodes[lit.id] = lit
        self.edges.append(Edge(source=entity.id, target=lit.id, relation="is named"))
        return entity

    def add_edge(self, source: Node, target: Node, relation: str) -> None:
        """Add a relation edge between two existing nodes."""
        self.edges.append(Edge(source=source.id, target=target.id, relation=relation))


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
# Graph I/O
# ---------------------------------------------------------------------------


def load_graphs(
    graphs_dir: Path,
) -> tuple[list[Graph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Load per-article graph JSON files from a directory.

    Returns:
        graphs: list of Graph objects
        entity_occurrences: entity_id -> list of {article_id, entity_id, name}
        edge_articles: (graph_id, src, tgt, relation) -> list of article_ids
    """
    entity_occurrences: dict[str, list[dict]] = {}
    edge_articles: dict[tuple, list[str]] = {}
    graphs: list[Graph] = []

    for path in sorted(graphs_dir.glob("*.json")):
        with open(path) as f:
            g = json.load(f)

        graph_id = g["id"]
        nodes: dict[str, Node] = {}

        for n in g["nodes"]:
            nid = n["id"]
            if "label" in n:
                nodes[nid] = LiteralNode(id=nid, graph_id=graph_id, label=n["label"])
            else:
                nodes[nid] = Node(id=nid, graph_id=graph_id)
                entity_occurrences[nid] = n["occurrences"]

        edges: list[Edge] = []
        for ed in g["edges"]:
            edge = Edge(
                source=ed["source"], target=ed["target"], relation=ed["relation"]
            )
            edges.append(edge)
            edge_articles[(graph_id, edge.source, edge.target, edge.relation)] = ed[
                "articles"
            ]

        graph = Graph(id=graph_id, nodes=nodes, edges=edges)
        graphs.append(graph)

    return graphs, entity_occurrences, edge_articles


def save_graphs(
    graphs: list[Graph],
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
    path: Path,
) -> None:
    """Write graphs + provenance to graph JSON format."""
    output_graphs = []
    for g in graphs:
        nodes_out = []
        for n in g.nodes.values():
            if isinstance(n, LiteralNode):
                nodes_out.append({"id": n.id, "label": n.label})
            else:
                nodes_out.append({"id": n.id, "occurrences": entity_occurrences[n.id]})
        edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "articles": edge_articles[
                    (g.id, edge.source, edge.target, edge.relation)
                ],
            }
            for edge in g.edges
        ]
        output_graphs.append({"id": g.id, "nodes": nodes_out, "edges": edges})

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"graphs": output_graphs}, f, indent=2)


# ---------------------------------------------------------------------------
# Embeddings and relation functionality
# ---------------------------------------------------------------------------


def embed(texts: list[str], model: TextEmbedding) -> dict[str, np.ndarray]:
    """Embed a list of texts, return text -> unit vector."""
    if not texts:
        return {}
    vecs = list(model.embed(texts))
    return {t: np.array(v) for t, v in zip(texts, vecs)}


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

    # Build entity-to-names mapping: for each entity, find its "is named" labels
    def _entity_names(g: Graph, eid: str) -> list[str]:
        names = []
        for edge in g.edges:
            if edge.relation == "is named" and edge.source == eid:
                tgt = g.nodes.get(edge.target)
                if isinstance(tgt, LiteralNode):
                    names.append(tgt.label)
        return names if names else [eid]

    # Collect all (src_name, tgt_name) pairs per relation phrase
    phrase_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for g in graphs:
        for edge in g.edges:
            for src_name in _entity_names(g, edge.source):
                for tgt_name in _entity_names(g, edge.target):
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


def prepare_embeddings(
    graphs: list[Graph],
    model: TextEmbedding,
    threshold: float = 0.8,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Functionality]]:
    """Embed all literal node labels and relation phrases; compute functionality weights.

    Returns (literal_embeddings, relation_embeddings, functionality).
    """
    all_literals = sorted({
        n.label for g in graphs for n in g.nodes.values()
        if isinstance(n, LiteralNode)
    })
    all_relations = sorted({edge.relation for g in graphs for edge in g.edges})
    literal_embeddings = embed(all_literals, model)
    # Wrap relation phrases as "A {phrase} B" to give the model syntactic context
    wrapped = [f"A {r} B" for r in all_relations]
    relation_embeddings = embed(wrapped, model)
    relation_embeddings = {
        r: relation_embeddings[w] for r, w in zip(all_relations, wrapped)
    }

    functionality = compute_functionality(graphs, relation_embeddings, threshold)

    return literal_embeddings, relation_embeddings, functionality


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


def propagate(
    graph_a: Graph,
    graph_b: Graph,
    literal_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    max_iter: int = 30,
    epsilon: float = 1e-4,
    rel_gate: float = 0.8,
    confidence_gate: float = 0.5,
    exp_lambda: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Run similarity propagation between two graphs.

    Entity names are literal nodes connected by "is named" edges, so
    name similarity flows through the same exponential-sum aggregation
    as any other structural evidence.  Literal-literal confidence is
    fixed (embedding cosine sim); entity-entity confidence is updated
    iteratively.

    Returns confidence: (node_id_a, node_id_b) -> float in [0, 1].
    """
    all_ids_a = list(graph_a.nodes)
    all_ids_b = list(graph_b.nodes)

    adj_a = _build_adjacency(graph_a, functionality)
    adj_b = _build_adjacency(graph_b, functionality)

    # Precompute which relation pairs pass the gate
    rel_passes_gate: set[tuple[str, str]] = set()
    rels_a = {edge.relation for edge in graph_a.edges}
    rels_b = {edge.relation for edge in graph_b.edges}
    for ra in rels_a:
        emb_a = relation_embeddings.get(ra)
        if emb_a is None:
            continue
        for rb in rels_b:
            emb_b = relation_embeddings.get(rb)
            if emb_b is None:
                continue
            if float(np.dot(emb_a, emb_b)) >= rel_gate:
                rel_passes_gate.add((ra, rb))

    # Initialize confidence for all node pairs
    confidence: dict[tuple[str, str], float] = {}
    literal_pairs: set[tuple[str, str]] = set()

    for id_a in all_ids_a:
        node_a = graph_a.nodes[id_a]
        for id_b in all_ids_b:
            node_b = graph_b.nodes[id_b]
            is_lit_a = isinstance(node_a, LiteralNode)
            is_lit_b = isinstance(node_b, LiteralNode)
            if is_lit_a and is_lit_b:
                # Literal-literal: fixed embedding cosine sim
                emb_a = literal_embeddings.get(node_a.label)
                emb_b = literal_embeddings.get(node_b.label)
                if emb_a is not None and emb_b is not None:
                    confidence[(id_a, id_b)] = max(0.0, float(np.dot(emb_a, emb_b)))
                else:
                    confidence[(id_a, id_b)] = 0.0
                literal_pairs.add((id_a, id_b))
            elif is_lit_a != is_lit_b:
                # Cross-type: always 0
                confidence[(id_a, id_b)] = 0.0
                literal_pairs.add((id_a, id_b))  # never updated
            else:
                # Entity-entity: starts at 0, updated by propagation
                confidence[(id_a, id_b)] = 0.0

    for _ in range(max_iter):
        changed = False

        for id_a in all_ids_a:
            for id_b in all_ids_b:
                if (id_a, id_b) in literal_pairs:
                    continue

                # Exponential sum over all qualifying edge pairs
                strength_sum = 0.0

                for nbr_a, rel_a, func_a in adj_a.get(id_a, []):
                    for nbr_b, rel_b, func_b in adj_b.get(id_b, []):
                        if (rel_a, rel_b) not in rel_passes_gate:
                            continue
                        nbr_conf = confidence[(nbr_a, nbr_b)]
                        if nbr_conf < confidence_gate:
                            continue
                        func_w = min(func_a, func_b)
                        strength_sum += func_w * nbr_conf

                combined = (
                    1.0 - math.exp(-exp_lambda * strength_sum)
                    if strength_sum > 0
                    else 0.0
                )

                old = confidence[(id_a, id_b)]
                if combined > old + epsilon:
                    confidence[(id_a, id_b)] = combined
                    changed = True

        if not changed:
            break

    return confidence


def select_matches(
    confidence: dict[tuple[str, str], float],
    ids_a: list[str],
    ids_b: list[str],
    threshold: float,
) -> list[tuple[str, str]]:
    """Select entity matches: pairs where confidence >= threshold."""
    return [
        (a, b)
        for a in ids_a
        for b in ids_b
        if confidence.get((a, b), 0.0) >= threshold
    ]


def match_round(
    graphs: list[Graph],
    literal_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    threshold: float = 0.8,
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> UnionFind:
    """One pass of pairwise propagation across all graphs.

    Returns a UnionFind with matched entity pairs keyed as (graph_id, entity_id).
    """
    uf = UnionFind()
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            confidence = propagate(
                graphs[i],
                graphs[j],
                literal_embeddings,
                relation_embeddings,
                functionality,
                max_iter=max_iter,
                epsilon=epsilon,
                rel_gate=threshold,
            )
            entity_ids_i = [
                nid for nid, n in graphs[i].nodes.items()
                if not isinstance(n, LiteralNode)
            ]
            entity_ids_j = [
                nid for nid, n in graphs[j].nodes.items()
                if not isinstance(n, LiteralNode)
            ]
            matches = select_matches(
                confidence,
                entity_ids_i,
                entity_ids_j,
                threshold=threshold,
            )
            for eid_a, eid_b in matches:
                uf.union((graphs[i].id, eid_a), (graphs[j].id, eid_b))
    return uf


# ---------------------------------------------------------------------------
# Graph merging
# ---------------------------------------------------------------------------


def merge_graphs(
    graphs: list[Graph],
    uf: UnionFind,
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
) -> tuple[list[Graph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Merge graphs whose entities have been linked by the union-find.

    Graphs with no matched entities are passed through unchanged.
    Merged entities keep all name variants as separate literal nodes;
    duplicate labels from different source graphs are deduplicated.
    Returns (new_graphs, new_entity_occurrences, new_edge_articles).
    """
    # Group graphs into merge components via shared entity roots
    graph_uf = UnionFind()
    for g in graphs:
        for n in g.nodes.values():
            if isinstance(n, LiteralNode):
                continue
            root = uf.find((g.id, n.id))
            if root != (g.id, n.id):
                root_graph_id = root[0]
                if root_graph_id != g.id:
                    graph_uf.union(g.id, root_graph_id)

    components: dict[str, list[Graph]] = defaultdict(list)
    for g in graphs:
        components[graph_uf.find(g.id)].append(g)

    new_graphs: list[Graph] = []
    new_entity_occ: dict[str, list[dict]] = {}
    new_edge_art: dict[tuple, list[str]] = {}

    for _root, component_graphs in components.items():
        if len(component_graphs) == 1:
            g = component_graphs[0]
            new_graphs.append(g)
            for n in g.nodes.values():
                if not isinstance(n, LiteralNode):
                    new_entity_occ[n.id] = entity_occurrences[n.id]
            for edge in g.edges:
                key = (g.id, edge.source, edge.target, edge.relation)
                new_edge_art[key] = edge_articles[key]
            continue

        merged_id = str(uuid.uuid4())

        # Group entity nodes by their union-find root
        entity_groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
        for g in component_graphs:
            for n in g.nodes.values():
                if isinstance(n, LiteralNode):
                    continue
                entity_groups[uf.find((g.id, n.id))].append((g.id, n.id))

        # Create merged entities: pool occurrences
        old_to_new: dict[tuple[str, str], str] = {}
        merged_nodes: dict[str, Node] = {}

        for _uf_root, members in entity_groups.items():
            new_eid = str(uuid.uuid4())
            pooled_occ: list[dict] = []
            for graph_id, entity_id in members:
                for occ in entity_occurrences[entity_id]:
                    pooled_occ.append(occ)
                old_to_new[(graph_id, entity_id)] = new_eid
            merged_nodes[new_eid] = Node(id=new_eid, graph_id=merged_id)
            new_entity_occ[new_eid] = pooled_occ

        # Collect literal nodes: deduplicate by label across source graphs
        # and map old literal IDs to new ones
        label_to_new_id: dict[str, str] = {}
        for g in component_graphs:
            for n in g.nodes.values():
                if isinstance(n, LiteralNode):
                    if n.label not in label_to_new_id:
                        new_lid = str(uuid.uuid4())
                        label_to_new_id[n.label] = new_lid
                        merged_nodes[new_lid] = LiteralNode(
                            id=new_lid, graph_id=merged_id, label=n.label
                        )
                    old_to_new[(g.id, n.id)] = label_to_new_id[n.label]

        # Remap edges, pool article provenance, pick most common relation phrase
        edge_pool: dict[tuple[str, str], list[str]] = defaultdict(list)
        edge_phrases: dict[tuple[str, str], list[str]] = defaultdict(list)
        for g in component_graphs:
            for edge in g.edges:
                new_src = old_to_new.get((g.id, edge.source))
                new_tgt = old_to_new.get((g.id, edge.target))
                if new_src is None or new_tgt is None:
                    continue
                pool_key = (new_src, new_tgt)
                old_key = (g.id, edge.source, edge.target, edge.relation)
                edge_pool[pool_key].extend(edge_articles.get(old_key, []))
                edge_phrases[pool_key].append(edge.relation)

        merged_edges: list[Edge] = []
        for (src, tgt), articles in edge_pool.items():
            phrase_counts: dict[str, int] = defaultdict(int)
            for p in edge_phrases[(src, tgt)]:
                phrase_counts[p] += 1
            best_relation = max(phrase_counts, key=phrase_counts.get)
            merged_edges.append(Edge(source=src, target=tgt, relation=best_relation))
            new_edge_art[(merged_id, src, tgt, best_relation)] = sorted(set(articles))

        merged = Graph(id=merged_id, nodes=merged_nodes, edges=merged_edges)
        new_graphs.append(merged)

    return new_graphs, new_entity_occ, new_edge_art


def match_all(
    graphs: list[Graph],
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
    threshold: float = 0.8,
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> tuple[list[Graph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Iterative match-merge loop until no new merges occur.

    Each round: embed, propagate, merge. Repeat because merged graphs have
    richer structure that may trigger new matches.

    Returns final (graphs, entity_occurrences, edge_articles).
    """
    model = TextEmbedding(model_name=os.environ["EMBEDDING_MODEL"])
    round_num = 0

    while True:
        round_num += 1
        n_before = len(graphs)

        click.echo(f"\n--- Round {round_num} ({n_before} graphs) ---")

        literal_embeddings, relation_embeddings, functionality = prepare_embeddings(
            graphs, model, threshold
        )
        uf = match_round(
            graphs, literal_embeddings, relation_embeddings, functionality,
            threshold=threshold, max_iter=max_iter, epsilon=epsilon,
        )
        graphs, entity_occurrences, edge_articles = merge_graphs(
            graphs, uf, entity_occurrences, edge_articles
        )

        click.echo(f"  {n_before} → {len(graphs)} graphs")

        if len(graphs) == n_before:
            break

    return graphs, entity_occurrences, edge_articles


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
    """Load graphs, run iterative match-merge, save."""
    graphs, entity_occurrences, edge_articles = load_graphs(graphs_dir)
    n_initial = len(graphs)
    click.echo(f"Loaded {n_initial} graphs from {graphs_dir}/")
    for g in graphs:
        entities = [n for n in g.nodes.values() if not isinstance(n, LiteralNode)]
        n_occ = sum(len(entity_occurrences[n.id]) for n in entities)
        click.echo(
            f"  {g.id}: {len(entities)} entities ({n_occ} occurrences), {len(g.edges)} edges"
        )

    graphs, entity_occurrences, edge_articles = match_all(
        graphs, entity_occurrences, edge_articles,
        threshold=threshold, max_iter=max_iter, epsilon=epsilon,
    )

    save_graphs(graphs, entity_occurrences, edge_articles, output_path)

    matched_entities = [
        (n, entity_occurrences[n.id])
        for g in graphs
        for n in g.nodes.values()
        if not isinstance(n, LiteralNode) and len(entity_occurrences[n.id]) > 1
    ]
    confirmed_edges = [
        (g, edge, edge_articles[(g.id, edge.source, edge.target, edge.relation)])
        for g in graphs
        for edge in g.edges
        if len(edge_articles.get((g.id, edge.source, edge.target, edge.relation), [])) > 1
    ]

    click.echo(f"\n{len(graphs)} graphs after merging (was {n_initial})")

    def _names(node_id: str, graph: Graph) -> str:
        if node_id in entity_occurrences:
            return " / ".join(sorted(set(o["name"] for o in entity_occurrences[node_id])))
        node = graph.nodes.get(node_id)
        return node.label if isinstance(node, LiteralNode) else node_id

    if matched_entities:
        click.echo(f"\n{len(matched_entities)} matched entities:")
        for n, occs in matched_entities:
            click.echo(f"  {_names(n.id, None)} — {len(occs)} occurrences")

    if confirmed_edges:
        click.echo(f"\n{len(confirmed_edges)} confirmed edges:")
        for g, edge, arts in confirmed_edges:
            click.echo(f"  {_names(edge.source, g)} —[{edge.relation}]→ {_names(edge.target, g)} ({len(arts)} sources)")

    click.echo(f"\nWrote {output_path}")
