import json
import uuid
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


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------


def load_graphs(
    path: Path,
) -> tuple[list[ArticleGraph], dict[str, list[dict]], dict[tuple, list[str]], dict[str, str]]:
    """Load graph JSON into ArticleGraph objects + provenance.

    Returns:
        graphs: list of ArticleGraph (graph id used as article_id)
        entity_occurrences: entity_id → list of {article_id, entity_id, name}
        edge_articles: (graph_id, src, tgt, cluster_id) → list of article_ids
        cluster_labels: cluster_id (str) → representative label
    """
    with open(path) as f:
        data = json.load(f)

    cluster_labels = data["cluster_labels"]
    entity_occurrences: dict[str, list[dict]] = {}
    edge_articles: dict[tuple, list[str]] = {}
    graphs: list[ArticleGraph] = []

    for g in data["graphs"]:
        graph_id = g["id"]
        entities: dict[str, Entity] = {}

        for e in g["entities"]:
            eid = e["id"]
            entities[eid] = Entity(
                id=eid, name=e["name"], type=e["type"], article_id=graph_id
            )
            entity_occurrences[eid] = e["occurrences"]

        edges: list[Edge] = []
        for ed in g["edges"]:
            edge = Edge(source=ed["source"], target=ed["target"], cluster_id=ed["cluster_id"])
            edges.append(edge)
            edge_articles[(graph_id, edge.source, edge.target, edge.cluster_id)] = ed["articles"]

        graphs.append(ArticleGraph(article_id=graph_id, entities=entities, edges=edges))

    return graphs, entity_occurrences, edge_articles, cluster_labels


def save_graphs(
    graphs: list[ArticleGraph],
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
    cluster_labels: dict[str, str],
    path: Path,
) -> None:
    """Write graphs + provenance to graph JSON format."""
    output_graphs = []
    for g in graphs:
        entities = []
        for e in g.entities.values():
            entities.append(
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "occurrences": entity_occurrences[e.id],
                }
            )

        edges = []
        for edge in g.edges:
            key = (g.article_id, edge.source, edge.target, edge.cluster_id)
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "cluster_id": edge.cluster_id,
                    "articles": edge_articles[key],
                }
            )

        output_graphs.append({"id": g.article_id, "entities": entities, "edges": edges})

    output = {"cluster_labels": cluster_labels, "graphs": output_graphs}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Entity embedding & comparison
# ---------------------------------------------------------------------------


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


def entity_embed_key(e: Entity) -> str:
    """Build the embedding key for an entity: '{type}: {name}'."""
    return f"{e.type}: {e.name}"


def entity_compatible(
    e1: Entity,
    e2: Entity,
    name_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> bool:
    """Check if two entities could be the same: embedding similarity >= threshold."""
    emb1 = name_embeddings.get(entity_embed_key(e1))
    emb2 = name_embeddings.get(entity_embed_key(e2))
    if emb1 is None or emb2 is None:
        return False
    return cosine_similarity(emb1, emb2) >= threshold


# ---------------------------------------------------------------------------
# Compatibility graph & max clique matching
# ---------------------------------------------------------------------------


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
    min_edges: int = 1,
) -> PairMatch | None:
    """Find the maximum common subgraph between two article graphs."""
    nodes, adjacency = build_compatibility_graph(
        graph_a, graph_b, name_embeddings, threshold
    )

    if not nodes:
        return None

    clique = bron_kerbosch(adjacency, len(nodes))
    if len(clique) < min_edges:
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


# ---------------------------------------------------------------------------
# Graph merging
# ---------------------------------------------------------------------------


def merge_graphs(
    graphs: list[ArticleGraph],
    all_pair_matches: list[PairMatch],
    uf: UnionFind,
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
) -> tuple[list[ArticleGraph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Merge matched graphs into larger graphs, pass singletons through.

    Returns (new_graphs, new_entity_occurrences, new_edge_articles).
    """
    # Find connected components of graphs via matched entities
    graph_uf = UnionFind()
    for pm in all_pair_matches:
        if pm.aligned_edges:
            graph_uf.union(pm.article_a, pm.article_b)

    graph_by_id = {g.article_id: g for g in graphs}
    components: dict[str, list[ArticleGraph]] = defaultdict(list)
    for g in graphs:
        root = graph_uf.find(g.article_id)
        components[root].append(g)

    new_graphs: list[ArticleGraph] = []
    new_entity_occ: dict[str, list[dict]] = {}
    new_edge_art: dict[tuple, list[str]] = {}

    for root, component_graphs in components.items():
        if len(component_graphs) == 1:
            # Singleton — pass through unchanged
            g = component_graphs[0]
            new_graphs.append(g)
            for e in g.entities.values():
                new_entity_occ[e.id] = entity_occurrences[e.id]
            for edge in g.edges:
                key = (g.article_id, edge.source, edge.target, edge.cluster_id)
                new_edge_art[key] = edge_articles[key]
            continue

        # --- Merge component ---
        merged_id = str(uuid.uuid4())

        # 1. Group entities by UF root
        entity_groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
        for g in component_graphs:
            for e in g.entities.values():
                uf_root = uf.find((g.article_id, e.id))
                entity_groups[uf_root].append((g.article_id, e.id))

        # 2. Create merged entities
        old_to_new: dict[tuple[str, str], str] = {}  # (graph_id, entity_id) → new_id
        merged_entities: dict[str, Entity] = {}

        for uf_root, members in entity_groups.items():
            new_eid = str(uuid.uuid4())

            pooled_occ: list[dict] = []
            name_counts: dict[str, int] = defaultdict(int)
            entity_type = None
            for graph_id, entity_id in members:
                for occ in entity_occurrences[entity_id]:
                    pooled_occ.append(occ)
                    name_counts[occ["name"]] += 1
                entity_type = graph_by_id[graph_id].entities[entity_id].type
                old_to_new[(graph_id, entity_id)] = new_eid

            canonical_name = max(name_counts, key=name_counts.get)
            merged_entities[new_eid] = Entity(
                id=new_eid, name=canonical_name, type=entity_type, article_id=merged_id
            )
            new_entity_occ[new_eid] = pooled_occ

        # 3. Remap and dedup edges
        edge_pool: dict[tuple[str, str, int], list[str]] = defaultdict(list)
        for g in component_graphs:
            for edge in g.edges:
                new_src = old_to_new.get((g.article_id, edge.source))
                new_tgt = old_to_new.get((g.article_id, edge.target))
                if new_src is None or new_tgt is None:
                    continue
                old_key = (g.article_id, edge.source, edge.target, edge.cluster_id)
                articles = edge_articles.get(old_key, [])
                edge_pool[(new_src, new_tgt, edge.cluster_id)].extend(articles)

        merged_edges: list[Edge] = []
        for (src, tgt, cid), articles in edge_pool.items():
            deduped = sorted(set(articles))
            merged_edges.append(Edge(source=src, target=tgt, cluster_id=cid))
            new_edge_art[(merged_id, src, tgt, cid)] = deduped

        new_graphs.append(
            ArticleGraph(article_id=merged_id, entities=merged_entities, edges=merged_edges)
        )

    return new_graphs, new_entity_occ, new_edge_art


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_matching(
    input_path: Path,
    output_path: Path,
    threshold: float,
    min_edges: int = 1,
) -> None:
    """Run structural matching: load graphs, match pairs, merge, save."""
    # 1. Load
    graphs, entity_occurrences, edge_articles, cluster_labels = load_graphs(input_path)
    click.echo(f"Loaded {len(graphs)} graphs from {input_path}")
    for g in graphs:
        n_occ = sum(len(entity_occurrences[e.id]) for e in g.entities.values())
        click.echo(
            f"  {g.article_id[:12]}: {len(g.entities)} entities "
            f"({n_occ} occurrences), {len(g.edges)} edges"
        )

    # 2. Embed entity names
    all_names: set[str] = set()
    for g in graphs:
        for e in g.entities.values():
            all_names.add(entity_embed_key(e))
    sorted_names = sorted(all_names)

    click.echo(f"\nEmbedding {len(sorted_names)} unique entity keys...")
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embeddings = embed_entity_names(sorted_names, model)

    # 3. Match all pairs
    n_pairs = len(graphs) * (len(graphs) - 1) // 2
    click.echo(f"\nMatching {n_pairs} pairs (threshold={threshold}, min_edges={min_edges})...")

    uf = UnionFind()
    all_pair_matches: list[PairMatch] = []

    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            pm = match_article_pair(graphs[i], graphs[j], name_embeddings, threshold, min_edges)
            if pm and pm.aligned_edges:
                all_pair_matches.append(pm)
                for ea, eb in pm.aligned_edges:
                    uf.union(
                        (pm.article_a, ea.source),
                        (pm.article_b, eb.source),
                    )
                    uf.union(
                        (pm.article_a, ea.target),
                        (pm.article_b, eb.target),
                    )

    click.echo(f"  {len(all_pair_matches)} pairs with matches")

    # 4. Merge
    merged_graphs, merged_occ, merged_edges = merge_graphs(
        graphs, all_pair_matches, uf, entity_occurrences, edge_articles
    )

    # 5. Save
    save_graphs(merged_graphs, merged_occ, merged_edges, cluster_labels, output_path)

    # Summary
    matched_entities = []
    for g in merged_graphs:
        for e in g.entities.values():
            occs = merged_occ[e.id]
            if len(occs) > 1:
                matched_entities.append((e, occs))

    confirmed_edges = []
    for g in merged_graphs:
        for edge in g.edges:
            key = (g.article_id, edge.source, edge.target, edge.cluster_id)
            arts = merged_edges[key]
            if len(arts) > 1:
                confirmed_edges.append((g, edge, arts))

    click.echo(f"\n{len(merged_graphs)} graphs after merging (was {len(graphs)})")

    if matched_entities:
        click.echo(f"\n{len(matched_entities)} matched entities:")
        for e, occs in matched_entities:
            names = sorted(set(o["name"] for o in occs))
            click.echo(f"  {e.name} ({e.type}) — {len(occs)} occurrences: {', '.join(names)}")

    if confirmed_edges:
        click.echo(f"\n{len(confirmed_edges)} confirmed edges:")
        for g, edge, arts in confirmed_edges:
            src = g.entities[edge.source].name
            tgt = g.entities[edge.target].name
            label = cluster_labels.get(str(edge.cluster_id), "?")
            click.echo(f"  {src} —[{label}]→ {tgt} ({len(arts)} sources)")

    click.echo(f"\nWrote {output_path}")
