"""Unit tests for _build_adjacency direction semantics."""

from worldgraph.graph import Graph
from worldgraph.match import Functionality, _build_adjacency


def test_source_gets_inverse_functionality():
    """For edge src --r--> tgt, src's adjacency entry should carry
    inverse functionality (target match implies source match)."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    g.add_edge(apple, beats, "acquired")

    func = {
        "acquired": Functionality(forward=0.5, inverse=0.8),
        "is named": Functionality(1.0, 1.0),
    }
    adj = _build_adjacency(g, func)

    # Find the "acquired" entry in apple's adjacency list
    acq_entries = [(nbr, rel, w) for nbr, rel, w in adj[apple.id] if rel == "acquired"]
    assert len(acq_entries) == 1
    _, _, weight = acq_entries[0]
    assert weight == 0.8  # inverse


def test_target_gets_forward_functionality():
    """For edge src --r--> tgt, tgt's adjacency entry should carry
    forward functionality (source match implies target match)."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    g.add_edge(apple, beats, "acquired")

    func = {
        "acquired": Functionality(forward=0.5, inverse=0.8),
        "is named": Functionality(1.0, 1.0),
    }
    adj = _build_adjacency(g, func)

    # Find the "acquired" entry in beats's adjacency list
    acq_entries = [(nbr, rel, w) for nbr, rel, w in adj[beats.id] if rel == "acquired"]
    assert len(acq_entries) == 1
    _, _, weight = acq_entries[0]
    assert weight == 0.5  # forward


def test_unknown_relation_defaults_to_1():
    """Relations not in the functionality dict should default to 1.0 for both directions."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    g.add_edge(apple, beats, "unknown_rel")

    adj = _build_adjacency(g, {})

    src_entry = [(nbr, rel, w) for nbr, rel, w in adj[apple.id] if rel == "unknown_rel"]
    tgt_entry = [(nbr, rel, w) for nbr, rel, w in adj[beats.id] if rel == "unknown_rel"]
    assert src_entry[0][2] == 1.0
    assert tgt_entry[0][2] == 1.0
