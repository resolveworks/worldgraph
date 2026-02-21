import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import click
import numpy as np
from fastembed import TextEmbedding


@dataclass
class Entity:
    id: str
    name: str
    type: str
    article_id: str


@dataclass
class Edge:
    source: str  # entity id
    target: str  # entity id
    cluster_id: int


@dataclass
class ArticleGraph:
    article_id: str
    entities: dict[str, Entity]  # id -> Entity
    edges: list[Edge]


@dataclass
class PairMatch:
    """Result of matching two article graphs."""

    article_a: str
    article_b: str
    # Each aligned edge: ((edge_a_idx, edge_b_idx), implied entity mappings)
    aligned_edges: list[tuple[Edge, Edge]]
    # Entity mappings: (article_a entity id) -> (article_b entity id)
    entity_map: dict[str, str]


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


def embed_entity_names(
    names: list[str], model: TextEmbedding
) -> dict[str, np.ndarray]:
    """Embed unique entity names, return name -> embedding vector."""
    if not names:
        return {}
    embeddings = list(model.embed(names))
    return {name: np.array(emb) for name, emb in zip(names, embeddings)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def entity_compatible(
    e1: Entity,
    e2: Entity,
    name_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> bool:
    """Check if two entities could be the same: same type + name similarity >= threshold."""
    if e1.type != e2.type:
        return False
    emb1 = name_embeddings.get(e1.name)
    emb2 = name_embeddings.get(e2.name)
    if emb1 is None or emb2 is None:
        return False
    return cosine_similarity(emb1, emb2) >= threshold


def build_article_graph(
    article_data: dict, relation_map: dict[str, int]
) -> ArticleGraph:
    """Build a normalized ArticleGraph from extraction data."""
    article_id = article_data["article_id"]
    entity_ids = {e["id"] for e in article_data["entities"]}
    entities = {
        e["id"]: Entity(
            id=e["id"], name=e["name"], type=e["type"], article_id=article_id
        )
        for e in article_data["entities"]
    }

    edges = []
    for rel in article_data["relations"]:
        src, tgt = rel["source"], rel["target"]
        # Skip edges where source or target isn't a known entity ID
        if src not in entity_ids or tgt not in entity_ids:
            continue
        phrase = rel["relation"]
        cluster_id = relation_map.get(phrase)
        if cluster_id is None:
            continue
        edges.append(Edge(source=src, target=tgt, cluster_id=cluster_id))

    return ArticleGraph(article_id=article_id, entities=entities, edges=edges)


def build_compatibility_graph(
    graph_a: ArticleGraph,
    graph_b: ArticleGraph,
    name_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    """Build compatibility graph over edge pairs.

    Returns (nodes, adjacency) where each node is (edge_a_idx, edge_b_idx)
    and adjacency maps node index -> set of compatible node indices.
    """
    # Step 1: Find compatible edge pairs (same cluster, compatible endpoints)
    nodes: list[tuple[int, int]] = []
    for i, ea in enumerate(graph_a.edges):
        for j, eb in enumerate(graph_b.edges):
            if ea.cluster_id != eb.cluster_id:
                continue
            src_a = graph_a.entities[ea.source]
            src_b = graph_b.entities[eb.source]
            tgt_a = graph_a.entities[ea.target]
            tgt_b = graph_b.entities[eb.target]
            if not entity_compatible(src_a, src_b, name_embeddings, threshold):
                continue
            if not entity_compatible(tgt_a, tgt_b, name_embeddings, threshold):
                continue
            nodes.append((i, j))

    # Step 2: Build adjacency — two nodes are compatible if entity mappings don't conflict
    adjacency: dict[int, set[int]] = defaultdict(set)
    for ni in range(len(nodes)):
        for nj in range(ni + 1, len(nodes)):
            ia, ib = nodes[ni]  # edge indices in graphs a, b
            ja, jb = nodes[nj]

            ea1, eb1 = graph_a.edges[ia], graph_b.edges[ib]
            ea2, eb2 = graph_a.edges[ja], graph_b.edges[jb]

            if _mappings_consistent(ea1, eb1, ea2, eb2):
                adjacency[ni].add(nj)
                adjacency[nj].add(ni)

    return nodes, adjacency


def _mappings_consistent(ea1: Edge, eb1: Edge, ea2: Edge, eb2: Edge) -> bool:
    """Check that implied entity mappings from two edge-pairs don't conflict.

    If the same entity in article A maps to the same entity in article B
    (and vice versa), and no entity maps to two different entities, they're consistent.
    """
    # Collect all implied mappings: a_entity -> b_entity
    map_a_to_b: dict[str, str] = {}
    map_b_to_a: dict[str, str] = {}

    pairs = [
        (ea1.source, eb1.source),
        (ea1.target, eb1.target),
        (ea2.source, eb2.source),
        (ea2.target, eb2.target),
    ]

    for a_ent, b_ent in pairs:
        if a_ent in map_a_to_b:
            if map_a_to_b[a_ent] != b_ent:
                return False
        else:
            map_a_to_b[a_ent] = b_ent

        if b_ent in map_b_to_a:
            if map_b_to_a[b_ent] != a_ent:
                return False
        else:
            map_b_to_a[b_ent] = a_ent

    return True


def bron_kerbosch(
    adjacency: dict[int, set[int]], nodes_count: int
) -> list[int]:
    """Bron-Kerbosch with pivoting. Returns the maximum clique."""
    best_clique: list[int] = []

    def _bk(R: set[int], P: set[int], X: set[int]):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Pick pivot with most connections in P ∪ X
        union = P | X
        pivot = max(union, key=lambda u: len(adjacency.get(u, set()) & P))
        for v in P - adjacency.get(pivot, set()):
            neighbors = adjacency.get(v, set())
            _bk(R | {v}, P & neighbors, X & neighbors)
            P = P - {v}
            X = X | {v}

    all_nodes = set(range(nodes_count))
    _bk(set(), all_nodes, set())
    return best_clique


def match_article_pair(
    graph_a: ArticleGraph,
    graph_b: ArticleGraph,
    name_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> PairMatch | None:
    """Find the maximum common subgraph between two article graphs."""
    nodes, adjacency = build_compatibility_graph(
        graph_a, graph_b, name_embeddings, threshold
    )

    if not nodes:
        return None

    clique = bron_kerbosch(adjacency, len(nodes))
    if not clique:
        return None

    # Extract aligned edges and entity mappings from the clique
    aligned_edges = []
    entity_map: dict[str, str] = {}

    for node_idx in clique:
        ia, ib = nodes[node_idx]
        ea = graph_a.edges[ia]
        eb = graph_b.edges[ib]
        aligned_edges.append((ea, eb))
        entity_map[ea.source] = eb.source
        entity_map[ea.target] = eb.target

    return PairMatch(
        article_a=graph_a.article_id,
        article_b=graph_b.article_id,
        aligned_edges=aligned_edges,
        entity_map=entity_map,
    )


def run_matching(
    extractions_path: Path,
    clusters_path: Path,
    output_path: Path,
    threshold: float,
) -> None:
    """Run the full structural matching pipeline."""
    with open(extractions_path) as f:
        extractions = json.load(f)
    with open(clusters_path) as f:
        clusters_data = json.load(f)

    relation_map = clusters_data["relation_map"]
    # Build cluster_id -> representative label lookup
    cluster_labels = {c["id"]: c["representative"] for c in clusters_data["clusters"]}

    # Build article graphs
    click.echo(f"Building graphs for {len(extractions)} articles...")
    graphs = [build_article_graph(art, relation_map) for art in extractions]
    for g in graphs:
        click.echo(f"  {g.article_id}: {len(g.entities)} entities, {len(g.edges)} edges")

    # Collect all unique entity names and embed them
    all_names = set()
    for g in graphs:
        for e in g.entities.values():
            all_names.add(e.name)
    all_names = sorted(all_names)

    click.echo(f"\nEmbedding {len(all_names)} unique entity names...")
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embeddings = embed_entity_names(all_names, model)

    # Match all article pairs
    n_pairs = len(graphs) * (len(graphs) - 1) // 2
    click.echo(f"\nMatching {n_pairs} article pairs (threshold={threshold})...")

    uf = UnionFind()
    # Track which edges (as entity_group_pair + cluster_id) are confirmed by which articles
    all_pair_matches: list[PairMatch] = []

    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            pm = match_article_pair(graphs[i], graphs[j], name_embeddings, threshold)
            if pm and pm.aligned_edges:
                all_pair_matches.append(pm)
                # Union matched entities
                for ea, eb in pm.aligned_edges:
                    key_a_src = (pm.article_a, ea.source)
                    key_b_src = (pm.article_b, eb.source)
                    key_a_tgt = (pm.article_a, ea.target)
                    key_b_tgt = (pm.article_b, eb.target)
                    uf.union(key_a_src, key_b_src)
                    uf.union(key_a_tgt, key_b_tgt)

    click.echo(f"  {len(all_pair_matches)} pairs with matches")

    # Build entity groups from Union-Find
    entity_lookup: dict[str, Entity] = {}
    for g in graphs:
        for e in g.entities.values():
            entity_lookup[(g.article_id, e.id)] = e

    # Group by root
    groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for key in uf.parent:
        root = uf.find(key)
        groups[root].append(key)

    # Filter to groups with >1 occurrence and build output
    entity_matches = []
    # Map: root key -> index in entity_matches
    root_to_idx: dict[tuple, int] = {}

    for root, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        occurrences = []
        name_counts: dict[str, int] = defaultdict(int)
        entity_type = None
        for article_id, entity_id in sorted(members):
            ent = entity_lookup.get((article_id, entity_id))
            if ent is None:
                continue
            occurrences.append(
                {"article_id": article_id, "entity_id": entity_id, "name": ent.name}
            )
            name_counts[ent.name] += 1
            entity_type = ent.type

        if not occurrences:
            continue

        canonical_name = max(name_counts, key=name_counts.get)
        idx = len(entity_matches)
        root_to_idx[root] = idx
        entity_matches.append(
            {
                "canonical_name": canonical_name,
                "type": entity_type,
                "occurrences": occurrences,
            }
        )

    # Build matched triples: for each aligned edge across all pair matches,
    # group by (source_entity_group, target_entity_group, cluster_id)
    triple_key_articles: dict[tuple, set[str]] = defaultdict(set)

    for pm in all_pair_matches:
        for ea, eb in pm.aligned_edges:
            src_root = uf.find((pm.article_a, ea.source))
            tgt_root = uf.find((pm.article_a, ea.target))
            src_idx = root_to_idx.get(src_root)
            tgt_idx = root_to_idx.get(tgt_root)
            if src_idx is None or tgt_idx is None:
                continue
            key = (src_idx, tgt_idx, ea.cluster_id)
            triple_key_articles[key].add(pm.article_a)
            triple_key_articles[key].add(pm.article_b)

    matched_triples = []
    for (src_idx, tgt_idx, cluster_id), articles in sorted(triple_key_articles.items()):
        matched_triples.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_labels.get(cluster_id, "?"),
                "source_match": src_idx,
                "target_match": tgt_idx,
                "confirming_articles": sorted(articles),
                "source_count": len(articles),
            }
        )

    output = {
        "parameters": {"name_threshold": threshold},
        "entity_matches": entity_matches,
        "matched_triples": matched_triples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    click.echo(f"\n{len(entity_matches)} entity groups:")
    for i, em in enumerate(entity_matches):
        arts = [o["article_id"] for o in em["occurrences"]]
        click.echo(f"  [{i}] {em['canonical_name']} ({em['type']}) — {', '.join(arts)}")

    click.echo(f"\n{len(matched_triples)} matched triples:")
    for mt in matched_triples:
        src = entity_matches[mt["source_match"]]["canonical_name"]
        tgt = entity_matches[mt["target_match"]]["canonical_name"]
        click.echo(
            f"  {src} —[{mt['cluster_label']}]→ {tgt} "
            f"({mt['source_count']} sources: {', '.join(mt['confirming_articles'])})"
        )

    click.echo(f"\nWrote {output_path}")
