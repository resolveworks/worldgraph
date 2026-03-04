"""Unit tests for compute_functionality."""

import pytest

from worldgraph.constants import NAME_EDGE
from worldgraph.graph import Graph
from worldgraph.match import compute_functionality


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_to_one_functionality_is_1(embed_relation):
    """A relation that maps each source to exactly one target has functionality 1.0."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    google = g.add_entity("Google")
    youtube = g.add_entity("YouTube")
    g.add_edge(apple, beats, "acquired")
    g.add_edge(google, youtube, "acquired")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g], rel_embs)
    assert func["acquired"].forward == pytest.approx(1.0)


def test_fan_out_lowers_functionality(embed_relation):
    """One source mapping to two distinct targets halves functionality."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    shazam = g.add_entity("Shazam")
    g.add_edge(apple, beats, "acquired")
    g.add_edge(apple, shazam, "acquired")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g], rel_embs)
    assert func["acquired"].forward == pytest.approx(0.5)


def test_one_to_one_inv_functionality_is_1(embed_relation):
    """A relation that maps each target from exactly one source has inv_functionality 1.0."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    google = g.add_entity("Google")
    youtube = g.add_entity("YouTube")
    g.add_edge(apple, beats, "acquired")
    g.add_edge(google, youtube, "acquired")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g], rel_embs)
    assert func["acquired"].inverse == pytest.approx(1.0)


def test_fan_in_lowers_inv_functionality(embed_relation):
    """Two sources mapping to the same target halves inv_functionality."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    google = g.add_entity("Google")
    beats = g.add_entity("Beats")
    g.add_edge(apple, beats, "acquired")
    g.add_edge(google, beats, "acquired")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g], rel_embs)
    assert func["acquired"].inverse == pytest.approx(0.5)


def test_similar_phrases_pool_edges(embed_relation):
    """'acquired' and 'bought' are similar (cosine ~0.82), so their edges pool —
    raising out-degree and lowering functionality."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    google = g1.add_entity("Google")
    youtube = g1.add_entity("YouTube")
    g1.add_edge(apple1, beats, "acquired")
    g1.add_edge(google, youtube, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    shazam = g2.add_entity("Shazam")
    g2.add_edge(apple2, shazam, "bought")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        "bought": embed_relation("bought"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g1, g2], rel_embs)
    assert func["acquired"].forward < 1.0


def test_dissimilar_phrases_do_not_pool(embed_relation):
    """'acquired' and 'located in' are dissimilar, so their edges don't pool.
    'acquired' sees only its own one-to-one edges → functionality stays 1.0."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    g1.add_edge(apple1, beats, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    california = g2.add_entity("California")
    us = g2.add_entity("US")
    g2.add_edge(apple2, california, "located in")
    g2.add_edge(apple2, us, "located in")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        "located in": embed_relation("located in"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g1, g2], rel_embs)
    assert func["acquired"].forward == pytest.approx(1.0)


def test_same_entity_name_across_graphs_pools(embed_relation):
    """The same source name in two graphs counts as one source, so two targets
    from that name across graphs raise out-degree."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    g1.add_edge(apple1, beats, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    shazam = g2.add_entity("Shazam")
    g2.add_edge(apple2, shazam, "acquired")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        NAME_EDGE: embed_relation(NAME_EDGE),
    }
    func = compute_functionality([g1, g2], rel_embs)
    # Apple→{Beats, Shazam}: avg_out_degree = 2 → functionality = 0.5
    assert func["acquired"].forward == pytest.approx(0.5)
