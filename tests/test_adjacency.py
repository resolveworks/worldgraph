"""Unit tests for _build_weighted_adjacency direction semantics."""

from worldgraph.graph import Graph
from worldgraph.match import Functionality, _build_weighted_adjacency


def test_source_gets_inverse_functionality():
    """For edge source --r--> target, source's adjacency entry should carry
    inverse functionality (target match implies source match)."""
    graph = Graph(id="g1")
    apple = graph.add_entity("Apple")
    beats = graph.add_entity("Beats")
    graph.add_edge(apple, beats, "acquired")

    func = {
        "acquired": Functionality(forward=0.5, inverse=0.8),
        "is named": Functionality(1.0, 1.0),
    }
    adjacency = _build_weighted_adjacency(graph, func)

    acquired_entries = [
        neighbor for neighbor in adjacency[apple.id] if neighbor.relation == "acquired"
    ]
    assert len(acquired_entries) == 1
    assert acquired_entries[0].func_weight == 0.8  # inverse


def test_target_gets_forward_functionality():
    """For edge source --r--> target, target's adjacency entry should carry
    forward functionality (source match implies target match)."""
    graph = Graph(id="g1")
    apple = graph.add_entity("Apple")
    beats = graph.add_entity("Beats")
    graph.add_edge(apple, beats, "acquired")

    func = {
        "acquired": Functionality(forward=0.5, inverse=0.8),
        "is named": Functionality(1.0, 1.0),
    }
    adjacency = _build_weighted_adjacency(graph, func)

    acquired_entries = [
        neighbor for neighbor in adjacency[beats.id] if neighbor.relation == "acquired"
    ]
    assert len(acquired_entries) == 1
    assert acquired_entries[0].func_weight == 0.5  # forward


def test_unknown_relation_defaults_to_1():
    """Relations not in the functionality dict should default to 1.0 for both directions."""
    graph = Graph(id="g1")
    apple = graph.add_entity("Apple")
    beats = graph.add_entity("Beats")
    graph.add_edge(apple, beats, "unknown_rel")

    adjacency = _build_weighted_adjacency(graph, {})

    source_entries = [
        neighbor
        for neighbor in adjacency[apple.id]
        if neighbor.relation == "unknown_rel"
    ]
    target_entries = [
        neighbor
        for neighbor in adjacency[beats.id]
        if neighbor.relation == "unknown_rel"
    ]
    assert source_entries[0].func_weight == 1.0
    assert target_entries[0].func_weight == 1.0
