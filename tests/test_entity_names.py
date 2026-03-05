"""Unit tests for Node.name."""

from worldgraph.graph import Graph


def test_add_entity_sets_name():
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    assert apple.name == "Apple"
