"""Unit tests for entity_names."""

from worldgraph.graph import Graph, entity_names


def test_returns_name_from_literal_node():
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    names = entity_names(g, apple.id)
    assert names == ["Apple"]


def test_returns_id_for_unknown_entity():
    g = Graph(id="g1")
    names = entity_names(g, "nonexistent")
    assert names == ["nonexistent"]


def test_multiple_names():
    """An entity with two NAME_EDGE edges returns both names."""
    g = Graph(id="g1")
    entity = g.add_entity("Apple")
    # add_entity already created one name edge; add another manually
    from worldgraph.constants import NAME_EDGE
    from worldgraph.graph import LiteralNode, Edge
    import uuid

    lit = LiteralNode(id=str(uuid.uuid4()), graph_id=g.id, label="Apple Inc")
    g.nodes[lit.id] = lit
    g.edges.append(Edge(source=entity.id, target=lit.id, relation=NAME_EDGE))

    names = entity_names(g, entity.id)
    assert sorted(names) == ["Apple", "Apple Inc"]
