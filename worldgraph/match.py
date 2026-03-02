"""Stage 2: Entity alignment via similarity propagation.

Algorithm overview
------------------
For each pair of graphs (Gi, Gj):

1. Seed confidence[(ei, ej)] with name similarity (cosine of name embeddings).

2. Propagate structural evidence via noisy-OR (PARIS-style):
   each iteration, for each entity pair (ei, ej), accumulate evidence
   from all edge pairs whose relation phrases pass the relation gate
   (cosine similarity >= threshold ⟹ "same relation"):

       evidence = 1 - Π(1 - func_weight * confidence[(nbr_a, nbr_b)])
       confidence[(ei, ej)] = max(confidence[(ei, ej)], evidence)

   Confidence is monotonically non-decreasing — convergence is guaranteed
   (FLORA / Knaster-Tarski fixpoint).

3. Select matches: keep pairs where confidence exceeds a threshold.

4. Merge matched entity pairs transitively via union-find, pool provenance.

References:
- Suchanek, Abiteboul, Senellart. "PARIS." PVLDB 2011.
- Peng, Bonald, Suchanek. "FLORA." 2025.
"""

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import click
import numpy as np
from fastembed import TextEmbedding


class Functionality(NamedTuple):
    forward: float
    inverse: float


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
            src_name = g.entities[edge.source].name
            tgt_name = g.entities[edge.target].name
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
    threshold: float = 0.8,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Functionality]]:
    """Embed all entity names and relation phrases; compute functionality weights.

    Returns (name_embeddings, relation_embeddings, functionality).
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

    functionality = compute_functionality(graphs, relation_embeddings, threshold)

    return name_embeddings, relation_embeddings, functionality


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
    name_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray],
    functionality: dict[str, Functionality],
    max_iter: int = 30,
    epsilon: float = 1e-4,
    rel_gate: float = 0.8,
    confidence_gate: float = 0.8,
) -> dict[tuple[str, str], float]:
    """Run similarity propagation between two graphs.

    Confidence is seeded with name similarity (cosine of name embeddings).
    Each iteration, evidence from neighbor pairs accumulates via noisy-OR:

        evidence = 1 - Π(1 - func_weight * confidence[neighbor])
        confidence[pair] = max(confidence[pair], evidence)

    The product runs over all edge pairs where the relation phrases pass
    the relation gate (cosine >= rel_gate ⟹ "same relation"; below ⟹
    different relation, no propagation).

    Confidence is monotonically non-decreasing — convergence guaranteed
    (FLORA / Knaster-Tarski).

    Returns confidence: (entity_id_a, entity_id_b) -> float in [0, 1].
    """
    ids_a = list(graph_a.entities)
    ids_b = list(graph_b.entities)

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

    # Seed: name similarity, computed once and held fixed.
    name_sim: dict[tuple[str, str], float] = {}
    for eid_a in ids_a:
        name_a = graph_a.entities[eid_a].name
        emb_a = name_embeddings.get(name_a)
        for eid_b in ids_b:
            name_b = graph_b.entities[eid_b].name
            emb_b = name_embeddings.get(name_b)
            if emb_a is not None and emb_b is not None:
                name_sim[(eid_a, eid_b)] = max(0.0, float(np.dot(emb_a, emb_b)))
            else:
                name_sim[(eid_a, eid_b)] = 0.0

    confidence: dict[tuple[str, str], float] = dict(name_sim)

    for _ in range(max_iter):
        changed = False

        for eid_a in ids_a:
            for eid_b in ids_b:
                # Noisy-OR over all qualifying edge pairs: structural evidence
                complement_product = 1.0

                for nbr_a, rel_a, func_a in adj_a.get(eid_a, []):
                    for nbr_b, rel_b, func_b in adj_b.get(eid_b, []):
                        if (rel_a, rel_b) not in rel_passes_gate:
                            continue
                        nbr_conf = confidence[(nbr_a, nbr_b)]
                        if nbr_conf < confidence_gate:
                            continue
                        func_w = min(func_a, func_b)
                        path_strength = func_w * nbr_conf
                        complement_product *= 1.0 - path_strength

                structural = 1.0 - complement_product

                # Combine the fixed name seed with structural evidence
                # via noisy-OR — independent evidence sources compound.
                # Recompute each iteration (don't accumulate) to avoid
                # double-counting the same evidence paths.
                seed = name_sim[(eid_a, eid_b)]
                combined = 1.0 - (1.0 - seed) * (1.0 - structural)

                old = confidence[(eid_a, eid_b)]
                if combined > old + epsilon:
                    confidence[(eid_a, eid_b)] = combined
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
        new_graphs.append(merged)

    return new_graphs, new_entity_occ, new_edge_art


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
    """Load graphs, run similarity propagation, merge, save."""
    graphs, entity_occurrences, edge_articles = load_graphs(graphs_dir)
    click.echo(f"Loaded {len(graphs)} graphs from {graphs_dir}/")
    for g in graphs:
        n_occ = sum(len(entity_occurrences[e.id]) for e in g.entities.values())
        click.echo(
            f"  {g.id[:12]}: {len(g.entities)} entities ({n_occ} occurrences), {len(g.edges)} edges"
        )

    click.echo("\nEmbedding entity names and relation phrases...")
    name_embeddings, relation_embeddings, functionality = prepare_embeddings(
        graphs, threshold
    )

    n_pairs = len(graphs) * (len(graphs) - 1) // 2
    click.echo(
        f"\nPropagating similarities over {n_pairs} graph pairs (threshold={threshold})..."
    )

    uf = UnionFind()
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            confidence = propagate(
                graphs[i],
                graphs[j],
                name_embeddings,
                relation_embeddings,
                functionality,
                max_iter=max_iter,
                epsilon=epsilon,
                rel_gate=threshold,
                confidence_gate=threshold,
            )
            matches = select_matches(
                confidence,
                list(graphs[i].entities),
                list(graphs[j].entities),
                threshold=threshold,
            )
            for eid_a, eid_b in matches:
                uf.union((graphs[i].id, eid_a), (graphs[j].id, eid_b))

    merged_graphs, merged_occ, merged_edges = merge_graphs(
        graphs, uf, entity_occurrences, edge_articles
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
