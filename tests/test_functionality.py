"""Unit tests for compute_functionality."""

import pytest

from worldgraph.match import Entity, Edge, Graph, compute_functionality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph(graph_id: str, edges: list[tuple[str, str, str]]) -> Graph:
    """Build a Graph from (src_name, tgt_name, relation) triples.

    Entity IDs are derived from names so the same name always yields the same ID
    within a graph (duplicate names are de-duped into one entity).
    """
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_to_one_functionality_is_1(embed_phrase):
    """A relation that maps each source to exactly one target has functionality 1.0."""
    g = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Google", "YouTube", "acquired"),
        ],
    )
    func, _ = compute_functionality([g], {"acquired": embed_phrase("acquired")})
    assert func["acquired"] == pytest.approx(1.0)


def test_fan_out_lowers_functionality(embed_phrase):
    """One source mapping to two distinct targets halves functionality."""
    g = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Apple", "Shazam", "acquired"),
        ],
    )
    func, _ = compute_functionality([g], {"acquired": embed_phrase("acquired")})
    assert func["acquired"] == pytest.approx(0.5)


def test_one_to_one_inv_functionality_is_1(embed_phrase):
    """A relation that maps each target from exactly one source has inv_functionality 1.0."""
    g = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Google", "YouTube", "acquired"),
        ],
    )
    _, inv_func = compute_functionality([g], {"acquired": embed_phrase("acquired")})
    assert inv_func["acquired"] == pytest.approx(1.0)


def test_fan_in_lowers_inv_functionality(embed_phrase):
    """Two sources mapping to the same target halves inv_functionality."""
    g = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Google", "Beats", "acquired"),
        ],
    )
    _, inv_func = compute_functionality([g], {"acquired": embed_phrase("acquired")})
    assert inv_func["acquired"] == pytest.approx(0.5)


def test_similar_phrases_pool_edges(embed_phrase):
    """'acquired' and 'bought' are similar (cosine ~0.82), so their edges pool —
    raising out-degree and lowering functionality."""
    # "acquired" alone: Apple→Beats, Google→YouTube → functionality 1.0
    # pooled with "bought": Apple also→Shazam → avg out-degree > 1 → func < 1.0
    g1 = make_graph(
        "g1", [("Apple", "Beats", "acquired"), ("Google", "YouTube", "acquired")]
    )
    g2 = make_graph("g2", [("Apple", "Shazam", "bought")])
    rel_embs = {"acquired": embed_phrase("acquired"), "bought": embed_phrase("bought")}
    func, _ = compute_functionality([g1, g2], rel_embs)
    assert func["acquired"] < 1.0


def test_dissimilar_phrases_do_not_pool(embed_phrase):
    """'acquired' and 'located in' are dissimilar, so their edges don't pool.
    'acquired' sees only its own one-to-one edges → functionality stays 1.0."""
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph(
        "g2",
        [
            ("Apple", "California", "located in"),
            ("Apple", "US", "located in"),
        ],
    )
    rel_embs = {
        "acquired": embed_phrase("acquired"),
        "located in": embed_phrase("located in"),
    }
    func, _ = compute_functionality([g1, g2], rel_embs)
    assert func["acquired"] == pytest.approx(1.0)


def test_same_entity_name_across_graphs_pools(embed_phrase):
    """The same source name in two graphs counts as one source, so two targets
    from that name across graphs raise out-degree."""
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Shazam", "acquired")])
    func, _ = compute_functionality([g1, g2], {"acquired": embed_phrase("acquired")})
    # Apple→{Beats, Shazam}: avg_out_degree = 2 → functionality = 0.5
    assert func["acquired"] == pytest.approx(0.5)
