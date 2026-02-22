import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
from fastembed import TextEmbedding


@dataclass
class Entity:
    id: str
    name: str
    article_id: str


@dataclass
class Edge:
    source: str  # entity id
    target: str  # entity id
    relation: str


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
    aligned_edges: list[tuple[Edge, Edge]]


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
) -> tuple[list[ArticleGraph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Load graph JSON into ArticleGraph objects + provenance.

    Returns:
        graphs: list of ArticleGraph (graph id used as article_id)
        entity_occurrences: entity_id → list of {article_id, entity_id, name}
        edge_articles: (graph_id, src, tgt, relation) → list of article_ids
    """
    with open(path) as f:
        data = json.load(f)

    entity_occurrences: dict[str, list[dict]] = {}
    edge_articles: dict[tuple, list[str]] = {}
    graphs: list[ArticleGraph] = []

    for g in data["graphs"]:
        graph_id = g["id"]
        entities: dict[str, Entity] = {}

        for e in g["entities"]:
            eid = e["id"]
            entities[eid] = Entity(
                id=eid, name=e["name"], article_id=graph_id
            )
            entity_occurrences[eid] = e["occurrences"]

        edges: list[Edge] = []
        for ed in g["edges"]:
            edge = Edge(source=ed["source"], target=ed["target"], relation=ed["relation"])
            edges.append(edge)
            edge_articles[(graph_id, edge.source, edge.target, edge.relation)] = ed["articles"]

        graphs.append(ArticleGraph(article_id=graph_id, entities=entities, edges=edges))

    return graphs, entity_occurrences, edge_articles


def save_graphs(
    graphs: list[ArticleGraph],
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
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
                    "occurrences": entity_occurrences[e.id],
                }
            )

        edges = []
        for edge in g.edges:
            key = (g.article_id, edge.source, edge.target, edge.relation)
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "articles": edge_articles[key],
                }
            )

        output_graphs.append({"id": g.article_id, "entities": entities, "edges": edges})

    output = {"graphs": output_graphs}

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


def embed_relation_phrases(
    phrases: list[str], model: TextEmbedding
) -> dict[str, np.ndarray]:
    """Embed relation phrases, return phrase -> embedding vector.

    Wraps each phrase as "A {phrase} B" to give the model syntactic context.
    """
    if not phrases:
        return {}
    wrapped = [f"A {phrase} B" for phrase in phrases]
    embeddings = list(model.embed(wrapped))
    return {phrase: np.array(emb) for phrase, emb in zip(phrases, embeddings)}


def compute_relation_specificities(
    relation_embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute specificity score for each relation phrase.

    Specificity = average cosine distance to all other relations.
    A relation that is semantically far from everything else is rare/specific (score ~1).
    A relation that clusters tightly with many others is common (score ~0).
    With only one relation, specificity defaults to 1.0.
    """
    phrases = list(relation_embeddings.keys())
    if len(phrases) <= 1:
        return {p: 1.0 for p in phrases}

    specificities: dict[str, float] = {}
    for phrase in phrases:
        emb = relation_embeddings[phrase]
        distances = [
            1.0 - cosine_similarity(emb, relation_embeddings[other])
            for other in phrases
            if other != phrase
        ]
        specificities[phrase] = sum(distances) / len(distances)
    return specificities


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def entity_similarity(
    e1: Entity,
    e2: Entity,
    name_embeddings: dict[str, np.ndarray],
) -> float:
    """Cosine similarity between two entity name embeddings."""
    emb1 = name_embeddings.get(e1.name)
    emb2 = name_embeddings.get(e2.name)
    if emb1 is None or emb2 is None:
        return 0.0
    return cosine_similarity(emb1, emb2)


# ---------------------------------------------------------------------------
# Compatibility graph & connected-component matching
# ---------------------------------------------------------------------------


def build_compatibility_graph(
    graph_a: ArticleGraph,
    graph_b: ArticleGraph,
    name_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    relation_specificities: dict[str, float],
    rel_floor: float = 0.8,
) -> tuple[list[tuple[int, int]], dict[int, set[int]], list[tuple[float, float, float]]]:
    """Build compatibility graph over edge pairs.

    Returns (nodes, adjacency, similarities) where each node is
    (edge_a_idx, edge_b_idx), adjacency maps node index -> set of compatible
    node indices, and similarities[i] = (src_sim, tgt_sim, specificity) for
    node i. specificity is the avg semantic uniqueness of the two relation
    phrases — how far each sits from all other relations in the corpus.
    """
    # Step 1: Find compatible edge pairs (similar relation, no name floor — evidence decay handles that)
    nodes: list[tuple[int, int]] = []
    similarities: list[tuple[float, float, float]] = []
    for i, ea in enumerate(graph_a.edges):
        for j, eb in enumerate(graph_b.edges):
            rel_sim = cosine_similarity(
                relation_embeddings[ea.relation], relation_embeddings[eb.relation]
            )
            if rel_sim < rel_floor:
                continue
            src_a = graph_a.entities[ea.source]
            src_b = graph_b.entities[eb.source]
            tgt_a = graph_a.entities[ea.target]
            tgt_b = graph_b.entities[eb.target]
            src_sim = entity_similarity(src_a, src_b, name_embeddings)
            tgt_sim = entity_similarity(tgt_a, tgt_b, name_embeddings)
            specificity = (
                relation_specificities.get(ea.relation, 1.0)
                + relation_specificities.get(eb.relation, 1.0)
            ) / 2.0
            nodes.append((i, j))
            similarities.append((src_sim, tgt_sim, specificity))

    # Step 2: Build adjacency — two nodes are adjacent iff they share an entity
    # endpoint in graph_a or graph_b AND their implied mappings are consistent.
    adjacency: dict[int, set[int]] = defaultdict(set)
    for ni in range(len(nodes)):
        for nj in range(ni + 1, len(nodes)):
            ia, ib = nodes[ni]  # edge indices in graphs a, b
            ja, jb = nodes[nj]

            ea1, eb1 = graph_a.edges[ia], graph_b.edges[ib]
            ea2, eb2 = graph_a.edges[ja], graph_b.edges[jb]

            # Structural connectivity: share an entity endpoint in either graph
            ents_ni = {ea1.source, ea1.target, eb1.source, eb1.target}
            ents_nj = {ea2.source, ea2.target, eb2.source, eb2.target}
            if not (ents_ni & ents_nj):
                continue

            if _mappings_consistent(ea1, eb1, ea2, eb2):
                adjacency[ni].add(nj)
                adjacency[nj].add(ni)

    return nodes, adjacency, similarities


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


def find_structural_matches(
    graph_a: ArticleGraph,
    graph_b: ArticleGraph,
    name_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    relation_specificities: dict[str, float],
    threshold: float,
    rel_floor: float = 0.8,
    evidence_scale: float = 2.0,
) -> PairMatch | None:
    """Find structurally connected common subgraphs between two article graphs.

    The required name similarity for each entity pair decays as structural
    evidence accumulates:

        required = threshold * exp(-evidence_scale * total_evidence)

    where total_evidence = sum of specificity over all matched edges touching
    that entity. With zero evidence the full threshold is required; with strong,
    specific-relation evidence the name floor drops toward 0.
    """
    nodes, adjacency, similarities = build_compatibility_graph(
        graph_a, graph_b, name_embeddings, relation_embeddings,
        relation_specificities, rel_floor=rel_floor,
    )

    if not nodes:
        return None

    # Find connected components via BFS over the compatibility graph
    visited: set[int] = set()
    aligned_edges = []

    for start in range(len(nodes)):
        if start in visited:
            continue
        # BFS
        component: list[int] = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Accumulate specificity-weighted evidence per entity endpoint
        entity_evidence: dict[str, float] = defaultdict(float)
        for node_idx in component:
            ia, ib = nodes[node_idx]
            src_sim, tgt_sim, specificity = similarities[node_idx]
            ea = graph_a.edges[ia]
            eb = graph_b.edges[ib]
            entity_evidence[(ea.source, eb.source)] += specificity
            entity_evidence[(ea.target, eb.target)] += specificity

        # Check each entity pair: name_sim >= threshold * exp(-evidence_scale * evidence)
        component_ok = True
        for node_idx in component:
            ia, ib = nodes[node_idx]
            src_sim, tgt_sim, _ = similarities[node_idx]
            ea = graph_a.edges[ia]
            eb = graph_b.edges[ib]
            src_evidence = entity_evidence[(ea.source, eb.source)]
            tgt_evidence = entity_evidence[(ea.target, eb.target)]
            src_required = threshold * np.exp(-evidence_scale * src_evidence)
            tgt_required = threshold * np.exp(-evidence_scale * tgt_evidence)
            if src_sim < src_required or tgt_sim < tgt_required:
                component_ok = False
                break

        if not component_ok:
            continue

        for node_idx in component:
            ia, ib = nodes[node_idx]
            aligned_edges.append((graph_a.edges[ia], graph_b.edges[ib]))

    if not aligned_edges:
        return None

    return PairMatch(
        article_a=graph_a.article_id,
        article_b=graph_b.article_id,
        aligned_edges=aligned_edges,
    )


# ---------------------------------------------------------------------------
# Graph merging
# ---------------------------------------------------------------------------


def merge_graphs(
    graphs: list[ArticleGraph],
    all_pair_matches: list[PairMatch],
    uf: UnionFind,
    relation_uf: UnionFind,
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
                key = (g.article_id, edge.source, edge.target, edge.relation)
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
            for graph_id, entity_id in members:
                for occ in entity_occurrences[entity_id]:
                    pooled_occ.append(occ)
                    name_counts[occ["name"]] += 1
                old_to_new[(graph_id, entity_id)] = new_eid

            canonical_name = max(name_counts, key=name_counts.get)
            merged_entities[new_eid] = Entity(
                id=new_eid, name=canonical_name, article_id=merged_id
            )
            new_entity_occ[new_eid] = pooled_occ

        # 3. Remap and dedup edges (normalize relations via relation_uf)
        # Pool by (new_src, new_tgt, canonical_relation)
        edge_pool: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        # Track original relation phrases per pool key for frequency-based naming
        edge_rel_phrases: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for g in component_graphs:
            for edge in g.edges:
                new_src = old_to_new.get((g.article_id, edge.source))
                new_tgt = old_to_new.get((g.article_id, edge.target))
                if new_src is None or new_tgt is None:
                    continue
                canonical_rel = relation_uf.find(edge.relation)
                pool_key = (new_src, new_tgt, canonical_rel)
                old_key = (g.article_id, edge.source, edge.target, edge.relation)
                articles = edge_articles.get(old_key, [])
                edge_pool[pool_key].extend(articles)
                edge_rel_phrases[pool_key].append(edge.relation)

        merged_edges: list[Edge] = []
        for (src, tgt, canonical_rel), articles in edge_pool.items():
            # Pick most common original phrase as the relation name
            phrases = edge_rel_phrases[(src, tgt, canonical_rel)]
            phrase_counts: dict[str, int] = defaultdict(int)
            for p in phrases:
                phrase_counts[p] += 1
            best_rel = max(phrase_counts, key=phrase_counts.get)
            deduped = sorted(set(articles))
            merged_edges.append(Edge(source=src, target=tgt, relation=best_rel))
            new_edge_art[(merged_id, src, tgt, best_rel)] = deduped

        new_graphs.append(
            ArticleGraph(article_id=merged_id, entities=merged_entities, edges=merged_edges)
        )

    return new_graphs, new_entity_occ, new_edge_art


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_embeddings(
    graphs: list[ArticleGraph],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Load embedding model, embed all entity names and relation phrases.

    Returns (name_embeddings, relation_embeddings, relation_specificities).
    """
    all_names: set[str] = set()
    all_relations: set[str] = set()
    for g in graphs:
        for e in g.entities.values():
            all_names.add(e.name)
        for edge in g.edges:
            all_relations.add(edge.relation)

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embeddings = embed_entity_names(sorted(all_names), model)
    relation_embeddings = embed_relation_phrases(sorted(all_relations), model)
    relation_specificities = compute_relation_specificities(relation_embeddings)

    return name_embeddings, relation_embeddings, relation_specificities


def run_match_merge(
    graphs: list[ArticleGraph],
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
    name_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    relation_specificities: dict[str, float],
    threshold: float,
    rel_floor: float,
    evidence_scale: float,
) -> tuple[list[ArticleGraph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Match all graph pairs and merge — no I/O, no printing.

    Returns (merged_graphs, merged_entity_occurrences, merged_edge_articles).
    """
    uf = UnionFind()
    relation_uf = UnionFind()
    all_pair_matches: list[PairMatch] = []

    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            pm = find_structural_matches(
                graphs[i], graphs[j], name_embeddings, relation_embeddings,
                relation_specificities, threshold, rel_floor, evidence_scale,
            )
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
                    relation_uf.union(ea.relation, eb.relation)

    return merge_graphs(
        graphs, all_pair_matches, uf, relation_uf, entity_occurrences, edge_articles
    )


def run_matching(
    input_path: Path,
    output_path: Path,
    threshold: float,
    rel_floor: float = 0.8,
    evidence_scale: float = 2.0,
) -> None:
    """Run structural matching: load graphs, match pairs, merge, save."""
    # 1. Load
    graphs, entity_occurrences, edge_articles = load_graphs(input_path)
    click.echo(f"Loaded {len(graphs)} graphs from {input_path}")
    for g in graphs:
        n_occ = sum(len(entity_occurrences[e.id]) for e in g.entities.values())
        click.echo(
            f"  {g.article_id[:12]}: {len(g.entities)} entities "
            f"({n_occ} occurrences), {len(g.edges)} edges"
        )

    # 2. Embed
    click.echo(f"\nEmbedding entity names and relation phrases...")
    name_embeddings, relation_embeddings, relation_specificities = prepare_embeddings(graphs)

    # 3. Match and merge
    n_pairs = len(graphs) * (len(graphs) - 1) // 2
    click.echo(f"\nMatching {n_pairs} pairs (threshold={threshold}, evidence_scale={evidence_scale})...")

    merged_graphs, merged_occ, merged_edges = run_match_merge(
        graphs, entity_occurrences, edge_articles,
        name_embeddings, relation_embeddings, relation_specificities,
        threshold, rel_floor, evidence_scale,
    )

    # 4. Save
    save_graphs(merged_graphs, merged_occ, merged_edges, output_path)

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
            key = (g.article_id, edge.source, edge.target, edge.relation)
            arts = merged_edges[key]
            if len(arts) > 1:
                confirmed_edges.append((g, edge, arts))

    click.echo(f"\n{len(merged_graphs)} graphs after merging (was {len(graphs)})")

    if matched_entities:
        click.echo(f"\n{len(matched_entities)} matched entities:")
        for e, occs in matched_entities:
            names = sorted(set(o["name"] for o in occs))
            click.echo(f"  {e.name} — {len(occs)} occurrences: {', '.join(names)}")

    if confirmed_edges:
        click.echo(f"\n{len(confirmed_edges)} confirmed edges:")
        for g, edge, arts in confirmed_edges:
            src = g.entities[edge.source].name
            tgt = g.entities[edge.target].name
            click.echo(f"  {src} —[{edge.relation}]→ {tgt} ({len(arts)} sources)")

    click.echo(f"\nWrote {output_path}")
