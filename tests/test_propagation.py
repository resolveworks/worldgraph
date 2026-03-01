"""Layer 2 tests for similarity propagation.

These tests exercise the propagate() function on small graphs with real
embeddings.  Tests verify that:

- Identical/synonym relations with matching names produce structural evidence
- Dissimilar relations, weak neighbors, and many weak paths produce no
  structural evidence and no spurious matches (GH issue #1)
"""

import numpy as np

from worldgraph.match import (
    Entity,
    Edge,
    Graph,
    compute_functionality,
    propagate,
    select_matches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph(graph_id: str, edges: list[tuple[str, str, str]]) -> Graph:
    """Build a Graph from (src_name, tgt_name, relation) triples."""
    entities: dict[str, Entity] = {}
    edge_objs: list[Edge] = []
    for src_name, tgt_name, relation in edges:
        src_id = f"{graph_id}:{src_name}"
        tgt_id = f"{graph_id}:{tgt_name}"
        if src_id not in entities:
            entities[src_id] = Entity(id=src_id, name=src_name, graph_id=graph_id)
        if tgt_id not in entities:
            entities[tgt_id] = Entity(id=tgt_id, name=tgt_name, graph_id=graph_id)
        edge_objs.append(Edge(source=src_id, target=tgt_id, relation=relation))
    g = Graph(id=graph_id, entities=entities, edges=edge_objs)
    g.index_edges()
    return g


def embed_names(entities: list[str], model) -> dict[str, np.ndarray]:
    """Embed entity names using the session model."""
    vecs = list(model.embed(entities))
    return {name: np.array(v) for name, v in zip(entities, vecs)}


def run_propagation(
    graph_a: Graph,
    graph_b: Graph,
    embed_phrase,
    name_embeddings: dict[str, np.ndarray],
    relations: list[str],
    threshold: float = 0.8,
):
    """Convenience wrapper: embed relations, compute functionality, propagate."""
    rel_embs = {r: embed_phrase(r) for r in relations}
    func, inv_func = compute_functionality([graph_a, graph_b], rel_embs, threshold)
    name_sim, structural = propagate(
        graph_a,
        graph_b,
        name_embeddings,
        rel_embs,
        func,
        inv_func,
    )
    return name_sim, structural


# ---------------------------------------------------------------------------
# Structural reinforcement tests
# ---------------------------------------------------------------------------


def test_identical_names_and_relations_reinforce(embed_phrase):
    """Two graphs with the same entity names and identical relations:
    propagation should produce structural evidence for the correct pairs."""
    #   g1: Apple --acquired--> Beats
    #   g2: Apple --acquired--> Beats
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "acquired")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats"], model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired"]
    )

    # Correct pairs should have structural evidence > 0
    assert structural[("g1:Apple", "g2:Apple")] > 0
    assert structural[("g1:Beats", "g2:Beats")] > 0

    # Wrong pairs should have no structural evidence
    assert structural[("g1:Apple", "g2:Beats")] == 0.0
    assert structural[("g1:Beats", "g2:Apple")] == 0.0


def test_synonym_relations_propagate(embed_phrase):
    """Synonym relation phrases ('acquired' / 'purchased') should propagate
    structural evidence just like identical ones."""
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "purchased")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats"], model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired", "purchased"]
    )

    # Correct pairs should have structural evidence
    assert structural[("g1:Apple", "g2:Apple")] > 0
    assert structural[("g1:Beats", "g2:Beats")] > 0


# ---------------------------------------------------------------------------
# No spurious matches (GH issue #1)
# ---------------------------------------------------------------------------


def test_dissimilar_relations_do_not_propagate(embed_phrase):
    """Unrelated relation phrases ('acquired' vs 'located in') should not
    produce structural evidence or matches.

    g1: Apple --acquired--> Beats
    g2: Tokyo --located in--> Japan
    """
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Tokyo", "Japan", "located in")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Apple", "Beats", "Tokyo", "Japan"]
    name_embs = embed_names(all_names, model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired", "located in"]
    )

    # No structural evidence should exist
    for key in structural:
        assert structural[key] == 0.0, (
            f"Spurious structural evidence: {key} = {structural[key]}"
        )

    # No matches
    matches = select_matches(
        name_sim,
        structural,
        list(g1.entities),
        list(g2.entities),
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == [], f"Spurious matches found: {matches}"


def test_weak_neighbors_do_not_propagate(embed_phrase):
    """Even with identical relation phrases, propagation should not produce
    structural evidence when neighbor name similarity is low.

    g1: Apple --acquired--> Beats
    g2: Google --acquired--> YouTube
    """
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Google", "YouTube", "acquired")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Apple", "Beats", "Google", "YouTube"]
    name_embs = embed_names(all_names, model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired"]
    )

    # No structural evidence — all neighbor pairs have low name similarity
    for key in structural:
        assert structural[key] == 0.0, (
            f"Spurious structural evidence: {key} = {structural[key]}"
        )


def test_many_weak_paths_do_not_accumulate(embed_phrase):
    """Many unrelated edges should not produce structural evidence or matches.

    g1: Org --acquired--> Target, Org --funded--> Project, Org --hired--> Person
    g2: City --located in--> Country, City --borders--> River, City --hosts--> Event
    """
    g1 = make_graph(
        "g1",
        [
            ("Org", "Target", "acquired"),
            ("Org", "Project", "funded"),
            ("Org", "Person", "hired"),
        ],
    )
    g2 = make_graph(
        "g2",
        [
            ("City", "Country", "located in"),
            ("City", "River", "borders"),
            ("City", "Event", "hosts"),
        ],
    )

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = [
        "Org",
        "Target",
        "Project",
        "Person",
        "City",
        "Country",
        "River",
        "Event",
    ]
    name_embs = embed_names(all_names, model)

    relations = ["acquired", "funded", "hired", "located in", "borders", "hosts"]
    name_sim, structural = run_propagation(g1, g2, embed_phrase, name_embs, relations)

    matches = select_matches(
        name_sim,
        structural,
        list(g1.entities),
        list(g2.entities),
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == [], f"Spurious matches from accumulated weak paths: {matches}"
