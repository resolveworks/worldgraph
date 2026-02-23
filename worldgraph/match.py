"""Stage 2: Entity alignment via PARIS-style similarity propagation.

Algorithm overview
------------------
For each pair of graphs (Gi, Gj):

1. Initialise σ[(ei, ej)] = cosine_similarity(name_emb(ei), name_emb(ej))
   for all entity pairs across the two graphs.

2. Propagate: each iteration, a pair's score is reinforced by its neighbours:

       σ_new[(ei, ej)] = σ_init[(ei, ej)]
           + Σ  σ[(ei', ej')] * rel_sim(r, r') * w(r, r')

   summed over all edges ei -r-> ei' in Gi and ej -r'-> ej' in Gj
   (and their reverses). rel_sim is cosine similarity of relation phrase
   embeddings. w(r, r') is the functionality weight — how specific/rare the
   relation is, approximated as the inverse average out-degree of that relation
   across the corpus.

3. Normalise scores by the maximum after each iteration. Stop when the
   Euclidean norm of the change vector falls below epsilon.

4. Apply SelectThreshold: for each entity ei, compute relative similarities
   (its score with each ej as a fraction of its best score). Keep pairs
   where both directions exceed the threshold.

5. Merge matched entity pairs transitively via union-find, pool provenance.

Reference: Suchanek, Abiteboul, Senellart. "PARIS: Probabilistic Alignment of
Relations, Instances, and Schema." PVLDB 2011.
"""

import json
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
from fastembed import TextEmbedding
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    id: str
    name: str
    graph_id: str


@dataclass
class Edge:
    source: str  # entity id
    target: str  # entity id
    relation: str


@dataclass
class Graph:
    id: str
    entities: dict[str, Entity]  # id -> Entity
    edges: list[Edge]
    # Adjacency indices built lazily via index_edges()
    out_edges: dict[str, list[Edge]] = field(default_factory=dict)
    in_edges: dict[str, list[Edge]] = field(default_factory=dict)

    def index_edges(self) -> None:
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        for edge in self.edges:
            self.out_edges[edge.source].append(edge)
            self.in_edges[edge.target].append(edge)


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
) -> tuple[list[Graph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Load graph JSON.

    Returns:
        graphs: list of Graph objects
        entity_occurrences: entity_id -> list of {article_id, entity_id, name}
        edge_articles: (graph_id, src, tgt, relation) -> list of article_ids
    """
    with open(path) as f:
        data = json.load(f)

    entity_occurrences: dict[str, list[dict]] = {}
    edge_articles: dict[tuple, list[str]] = {}
    graphs: list[Graph] = []

    for g in data["graphs"]:
        graph_id = g["id"]
        entities: dict[str, Entity] = {}

        for e in g["entities"]:
            eid = e["id"]
            entities[eid] = Entity(id=eid, name=e["name"], graph_id=graph_id)
            entity_occurrences[eid] = e["occurrences"]

        edges: list[Edge] = []
        for ed in g["edges"]:
            edge = Edge(
                source=ed["source"], target=ed["target"], relation=ed["relation"]
            )
            edges.append(edge)
            edge_articles[(graph_id, edge.source, edge.target, edge.relation)] = ed[
                "articles"
            ]

        graph = Graph(id=graph_id, entities=entities, edges=edges)
        graph.index_edges()
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
        entities = [
            {"id": e.id, "name": e.name, "occurrences": entity_occurrences[e.id]}
            for e in g.entities.values()
        ]
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
        output_graphs.append({"id": g.id, "entities": entities, "edges": edges})

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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def cluster_relations(
    relation_embeddings: dict[str, np.ndarray],
    distance_threshold: float = 0.3,
) -> dict[str, int]:
    """Cluster relation phrases by semantic similarity using HAC with complete linkage.

    Embeddings should be the "A {phrase} B" wrapped vectors already computed in
    prepare_embeddings — not raw phrase embeddings. Complete linkage ensures all
    members of a cluster are mutually similar, producing compact synonym groups.
    Singletons (phrases that never merged) get their own cluster id.

    distance_threshold is in cosine distance space (= 1 - cosine_similarity),
    so 0.3 corresponds to cosine similarity ~0.7.
    """
    phrases = list(relation_embeddings)
    if len(phrases) == 0:
        return {}
    if len(phrases) == 1:
        return {phrases[0]: 0}

    matrix = np.stack([relation_embeddings[p] for p in phrases])
    dist_matrix = 1.0 - (matrix @ matrix.T)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="complete",
        compute_full_tree=True,
    )
    labels = clustering.fit_predict(dist_matrix)
    return {phrase: int(label) for phrase, label in zip(phrases, labels)}


def compute_functionality(
    graphs: list[Graph],
    relation_embeddings: dict[str, np.ndarray],
    distance_threshold: float = 0.3,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute functionality and inverse functionality for each relation phrase.

    Functionality ≈ 1 / avg_out_degree: for a given source name, how many
    distinct target names does it map to via this relation cluster? High means
    the relation uniquely determines the target — strong forward evidence.

    Inverse functionality ≈ 1 / avg_in_degree: for a given target name, how
    many distinct source names map to it via this relation cluster? High means
    the relation uniquely determines the source — strong backward evidence.

    Entity names (not IDs) are used so that the same entity mentioned across
    multiple graphs pools its statistics. Relation phrases are clustered by
    semantic similarity before computing degrees, so synonym phrasings share
    statistics.

    Returns (functionality, inverse_functionality), each a dict from phrase to float.
    """
    cluster_of = cluster_relations(relation_embeddings, distance_threshold)

    # cluster -> list of (source_name, target_name) pairs across all graphs
    cluster_pairs: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for g in graphs:
        for edge in g.edges:
            src_name = g.entities[edge.source].name
            tgt_name = g.entities[edge.target].name
            cid = cluster_of[edge.relation]
            cluster_pairs[cid].append((src_name, tgt_name))

    cluster_func: dict[int, float] = {}
    cluster_inv_func: dict[int, float] = {}
    for cid, pairs in cluster_pairs.items():
        targets_per_source: dict[str, set[str]] = defaultdict(set)
        sources_per_target: dict[str, set[str]] = defaultdict(set)
        for src, tgt in pairs:
            targets_per_source[src].add(tgt)
            sources_per_target[tgt].add(src)
        avg_out_degree = sum(len(v) for v in targets_per_source.values()) / len(
            targets_per_source
        )
        avg_in_degree = sum(len(v) for v in sources_per_target.values()) / len(
            sources_per_target
        )
        cluster_func[cid] = 1.0 / avg_out_degree
        cluster_inv_func[cid] = 1.0 / avg_in_degree

    functionality = {phrase: cluster_func[cid] for phrase, cid in cluster_of.items()}
    inv_functionality = {
        phrase: cluster_inv_func[cid] for phrase, cid in cluster_of.items()
    }
    return functionality, inv_functionality


def prepare_embeddings(
    graphs: list[Graph],
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, float]
]:
    """Embed all entity names and relation phrases; compute functionality weights.

    Returns (name_embeddings, relation_embeddings, functionality, inv_functionality).
    """
    all_names = sorted({e.name for g in graphs for e in g.entities.values()})
    all_relations = sorted({edge.relation for g in graphs for edge in g.edges})

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embeddings = embed(all_names, model)
    # Wrap relation phrases as "A {phrase} B" to give the model syntactic context
    wrapped = [f"A {r} B" for r in all_relations]
    relation_embeddings = embed(wrapped, model)
    relation_embeddings = {
        r: relation_embeddings[w] for r, w in zip(all_relations, wrapped)
    }

    functionality, inv_functionality = compute_functionality(
        graphs, relation_embeddings
    )

    return name_embeddings, relation_embeddings, functionality, inv_functionality


# ---------------------------------------------------------------------------
# Similarity propagation
# ---------------------------------------------------------------------------


def propagate(
    graph_a: Graph,
    graph_b: Graph,
    name_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, float],
    inv_functionality: dict[str, float],
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> dict[tuple[str, str], float]:
    """Run PARIS-style similarity propagation between two graphs.

    Returns a dict mapping (entity_id_a, entity_id_b) -> converged similarity.
    """
    ids_a = list(graph_a.entities)
    ids_b = list(graph_b.entities)

    # Initialise from name embedding similarity
    sigma: dict[tuple[str, str], float] = {}
    for eid_a in ids_a:
        name_a = graph_a.entities[eid_a].name
        emb_a = name_embeddings.get(name_a)
        for eid_b in ids_b:
            name_b = graph_b.entities[eid_b].name
            emb_b = name_embeddings.get(name_b)
            if emb_a is not None and emb_b is not None:
                sigma[(eid_a, eid_b)] = max(0.0, cosine_sim(emb_a, emb_b))
            else:
                sigma[(eid_a, eid_b)] = 0.0

    sigma_init = dict(sigma)

    for _ in range(max_iter):
        increment: dict[tuple[str, str], float] = defaultdict(float)

        # For every pair of edges (one from each graph) connected by similar relations,
        # propagate the similarity of their endpoints back to the source pair.
        for eid_a in ids_a:
            for eid_b in ids_b:
                # Outgoing edges: ei -r-> ei'  and  ej -r'-> ej'
                for edge_a in graph_a.out_edges.get(eid_a, []):
                    emb_r = relation_embeddings.get(edge_a.relation)
                    if emb_r is None:
                        continue
                    for edge_b in graph_b.out_edges.get(eid_b, []):
                        emb_r2 = relation_embeddings.get(edge_b.relation)
                        if emb_r2 is None:
                            continue
                        rel_sim = max(0.0, cosine_sim(emb_r, emb_r2))
                        if rel_sim == 0.0:
                            continue
                        w = (
                            rel_sim
                            * (
                                functionality.get(edge_a.relation, 1.0)
                                + functionality.get(edge_b.relation, 1.0)
                            )
                            / 2.0
                        )
                        neighbor_sim = sigma.get((edge_a.target, edge_b.target), 0.0)
                        increment[(eid_a, eid_b)] += neighbor_sim * w

                # Incoming edges: ei' -r-> ei  and  ej' -r'-> ej
                for edge_a in graph_a.in_edges.get(eid_a, []):
                    emb_r = relation_embeddings.get(edge_a.relation)
                    if emb_r is None:
                        continue
                    for edge_b in graph_b.in_edges.get(eid_b, []):
                        emb_r2 = relation_embeddings.get(edge_b.relation)
                        if emb_r2 is None:
                            continue
                        rel_sim = max(0.0, cosine_sim(emb_r, emb_r2))
                        if rel_sim == 0.0:
                            continue
                        w = (
                            rel_sim
                            * (
                                inv_functionality.get(edge_a.relation, 1.0)
                                + inv_functionality.get(edge_b.relation, 1.0)
                            )
                            / 2.0
                        )
                        neighbor_sim = sigma.get((edge_a.source, edge_b.source), 0.0)
                        increment[(eid_a, eid_b)] += neighbor_sim * w

        # σ_new = σ_init + increment, then normalise
        sigma_new: dict[tuple[str, str], float] = {}
        max_val = 0.0
        for key in sigma:
            val = sigma_init[key] + increment.get(key, 0.0)
            sigma_new[key] = val
            max_val = max(max_val, val)

        if max_val > 0:
            for key in sigma_new:
                sigma_new[key] /= max_val

        # Convergence check: Euclidean norm of change vector
        residual = math.sqrt(sum((sigma_new[k] - sigma[k]) ** 2 for k in sigma))
        sigma = sigma_new
        if residual < epsilon:
            break

    return sigma


def select_matches(
    sigma: dict[tuple[str, str], float],
    ids_a: list[str],
    ids_b: list[str],
    threshold: float,
) -> list[tuple[str, str]]:
    """Apply SelectThreshold to the converged similarity matrix.

    For each entity in A, compute relative similarities: normalise its scores
    against all B-entities by its best score. Keep pairs where both
    A-relative and B-relative similarity exceed the threshold.
    """
    # Relative similarity from A's perspective
    rel_a: dict[tuple[str, str], float] = {}
    for eid_a in ids_a:
        best = max((sigma.get((eid_a, eid_b), 0.0) for eid_b in ids_b), default=0.0)
        if best > 0:
            for eid_b in ids_b:
                rel_a[(eid_a, eid_b)] = sigma.get((eid_a, eid_b), 0.0) / best

    # Relative similarity from B's perspective
    rel_b: dict[tuple[str, str], float] = {}
    for eid_b in ids_b:
        best = max((sigma.get((eid_a, eid_b), 0.0) for eid_a in ids_a), default=0.0)
        if best > 0:
            for eid_a in ids_a:
                rel_b[(eid_a, eid_b)] = sigma.get((eid_a, eid_b), 0.0) / best

    matches = []
    for eid_a in ids_a:
        for eid_b in ids_b:
            if (
                rel_a.get((eid_a, eid_b), 0.0) >= threshold
                and rel_b.get((eid_a, eid_b), 0.0) >= threshold
            ):
                matches.append((eid_a, eid_b))

    return matches


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
    Returns (new_graphs, new_entity_occurrences, new_edge_articles).
    """
    # Group graphs into merge components via shared entity roots
    graph_uf = UnionFind()
    for g in graphs:
        for e in g.entities.values():
            root = uf.find((g.id, e.id))
            # If this entity was merged with something from another graph,
            # union those graphs together
            if root != (g.id, e.id):
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
            for e in g.entities.values():
                new_entity_occ[e.id] = entity_occurrences[e.id]
            for edge in g.edges:
                key = (g.id, edge.source, edge.target, edge.relation)
                new_edge_art[key] = edge_articles[key]
            continue

        merged_id = str(uuid.uuid4())

        # Group entities by their union-find root
        entity_groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
        for g in component_graphs:
            for e in g.entities.values():
                entity_groups[uf.find((g.id, e.id))].append((g.id, e.id))

        # Create merged entities: pool occurrences, pick most common name
        old_to_new: dict[tuple[str, str], str] = {}
        merged_entities: dict[str, Entity] = {}

        for _uf_root, members in entity_groups.items():
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
                id=new_eid, name=canonical_name, graph_id=merged_id
            )
            new_entity_occ[new_eid] = pooled_occ

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

        merged = Graph(id=merged_id, entities=merged_entities, edges=merged_edges)
        merged.index_edges()
        new_graphs.append(merged)

    return new_graphs, new_entity_occ, new_edge_art


# ---------------------------------------------------------------------------
# Top-level matching pipeline
# ---------------------------------------------------------------------------


def run_match_merge(
    graphs: list[Graph],
    entity_occurrences: dict[str, list[dict]],
    edge_articles: dict[tuple, list[str]],
    name_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, float],
    inv_functionality: dict[str, float],
    threshold: float,
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> tuple[list[Graph], dict[str, list[dict]], dict[tuple, list[str]]]:
    """Run propagation over all graph pairs, collect matches, merge.

    Returns (merged_graphs, merged_entity_occurrences, merged_edge_articles).
    """
    uf = UnionFind()

    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            sigma = propagate(
                graphs[i],
                graphs[j],
                name_embeddings,
                relation_embeddings,
                functionality,
                inv_functionality,
                max_iter=max_iter,
                epsilon=epsilon,
            )
            ids_a = list(graphs[i].entities)
            ids_b = list(graphs[j].entities)
            matches = select_matches(sigma, ids_a, ids_b, threshold)
            for eid_a, eid_b in matches:
                uf.union((graphs[i].id, eid_a), (graphs[j].id, eid_b))

    return merge_graphs(graphs, uf, entity_occurrences, edge_articles)


def run_matching(
    input_path: Path,
    output_path: Path,
    threshold: float,
    max_iter: int = 30,
    epsilon: float = 1e-4,
) -> None:
    """Load graphs, run similarity propagation, merge, save."""
    graphs, entity_occurrences, edge_articles = load_graphs(input_path)
    click.echo(f"Loaded {len(graphs)} graphs from {input_path}")
    for g in graphs:
        n_occ = sum(len(entity_occurrences[e.id]) for e in g.entities.values())
        click.echo(
            f"  {g.id[:12]}: {len(g.entities)} entities ({n_occ} occurrences), {len(g.edges)} edges"
        )

    click.echo("\nEmbedding entity names and relation phrases...")
    name_embeddings, relation_embeddings, functionality, inv_functionality = (
        prepare_embeddings(graphs)
    )

    n_pairs = len(graphs) * (len(graphs) - 1) // 2
    click.echo(
        f"\nPropagating similarities over {n_pairs} graph pairs (threshold={threshold})..."
    )

    merged_graphs, merged_occ, merged_edges = run_match_merge(
        graphs,
        entity_occurrences,
        edge_articles,
        name_embeddings,
        relation_embeddings,
        functionality,
        inv_functionality,
        threshold=threshold,
        max_iter=max_iter,
        epsilon=epsilon,
    )

    save_graphs(merged_graphs, merged_occ, merged_edges, output_path)

    matched_entities = [
        (e, merged_occ[e.id])
        for g in merged_graphs
        for e in g.entities.values()
        if len(merged_occ[e.id]) > 1
    ]
    confirmed_edges = [
        (g, edge, merged_edges[(g.id, edge.source, edge.target, edge.relation)])
        for g in merged_graphs
        for edge in g.edges
        if len(merged_edges[(g.id, edge.source, edge.target, edge.relation)]) > 1
    ]

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
