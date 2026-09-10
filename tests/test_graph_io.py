"""Tests for graph save/load round-trip: provenance, node kinds, multi-label
names, role edges, and validation of the event-source invariant."""

import json
from pathlib import Path

import pytest
from conftest import fact

from worldgraph.graph import Edge, Graph, Node, load_graph, save_graph


def test_save_load_roundtrip_single_graph(tmp_path: Path):
    """Single-article graph: graph_id is always serialized per node and edge."""
    g = Graph(id="article-1")
    alice = g.add_entity("Alice")
    bob = g.add_entity("Bob")
    fact(g, "know", agent=alice, patient=bob)

    path = tmp_path / "g.json"
    save_graph(g, path)

    with open(path) as f:
        data = json.load(f)
    for node_data in data["nodes"]:
        assert node_data["graph_id"] == "article-1"
    for edge_data in data["edges"]:
        assert edge_data["graph_id"] == "article-1"

    loaded = load_graph(path)
    for node in loaded.nodes.values():
        assert node.graph_id == "article-1"
    for edge in loaded.edges.values():
        assert edge.graph_id == "article-1"


def test_node_kind_roundtrip(tmp_path: Path):
    """Entity/event kinds and the event label survive save/load."""
    g = Graph(id="article-1")
    alice = g.add_entity("Alice")
    event = fact(g, "resign", agent=alice)

    path = tmp_path / "g.json"
    save_graph(g, path)

    with open(path) as f:
        data = json.load(f)
    kinds = {n["id"]: n["kind"] for n in data["nodes"]}
    assert kinds[alice.id] == "entity"
    assert kinds[event.id] == "event"

    loaded = load_graph(path)
    assert loaded.nodes[alice.id].kind == "entity"
    assert loaded.nodes[event.id].kind == "event"
    assert loaded.nodes[event.id].names == ["resign"]


def test_edge_role_roundtrip(tmp_path: Path):
    """Edges serialize their role, and event-to-event participation
    (join targeting the visit event) survives the round-trip."""
    g = Graph(id="article-1")
    ivo = g.add_entity("Ivo Brandt")
    visit = fact(g, "visit", agent=ivo)
    fact(g, "join", agent=ivo, patient=visit)

    path = tmp_path / "g.json"
    save_graph(g, path)
    loaded = load_graph(path)

    roles = sorted(edge.role for edge in loaded.edges.values())
    assert roles == ["agent", "agent", "patient"]
    join = loaded.nodes[next(n.id for n in g.nodes.values() if n.names == ["join"])]
    join_patient = next(
        e for e in loaded.edges.values() if e.source == join.id and e.role == "patient"
    )
    assert loaded.nodes[join_patient.target].names == ["visit"]


def test_save_load_roundtrip_unified_graph(tmp_path: Path):
    """Unified graph with terms from different source graphs preserves graph_id."""
    g = Graph(id="unified")
    g.nodes["n1"] = Node(id="n1", graph_id="article-1", names=["Alice"], kind="entity")
    g.nodes["n2"] = Node(id="n2", graph_id="article-2", names=["Bob"], kind="entity")
    g.nodes["n3"] = Node(id="n3", graph_id="article-1", names=["know"], kind="event")
    g.edges["x1"] = Edge(
        id="x1", graph_id="article-1", source="n3", target="n2", role="patient",
    )

    path = tmp_path / "unified.json"
    save_graph(g, path)

    with open(path) as f:
        data = json.load(f)
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert nodes_by_id["n1"]["graph_id"] == "article-1"
    assert nodes_by_id["n2"]["graph_id"] == "article-2"
    assert data["edges"][0]["graph_id"] == "article-1"

    loaded = load_graph(path)
    assert loaded.nodes["n1"].graph_id == "article-1"
    assert loaded.nodes["n2"].graph_id == "article-2"
    assert loaded.edges["x1"].graph_id == "article-1"


def test_save_load_roundtrip_multi_label_names(tmp_path: Path):
    """Entities with multiple names survive save/load round-trip."""
    g = Graph(id="article-1")
    n1 = g.add_entity(["Meridian Technologies", "Meridian Tech"])
    n2 = g.add_entity("DataVault")
    fact(g, "acquire", agent=n1, patient=n2)

    path = tmp_path / "g.json"
    save_graph(g, path)

    with open(path) as f:
        data = json.load(f)
    node_by_id = {n["id"]: n for n in data["nodes"]}
    assert node_by_id[n1.id]["names"] == ["Meridian Technologies", "Meridian Tech"]
    assert node_by_id[n2.id]["names"] == ["DataVault"]

    loaded = load_graph(path)
    assert loaded.nodes[n1.id].names == ["Meridian Technologies", "Meridian Tech"]
    assert loaded.nodes[n2.id].names == ["DataVault"]


def test_load_duplicate_node_id_raises(tmp_path: Path):
    """Duplicate node ids are invalid — no silent overwrite."""
    data = {
        "id": "article-1",
        "nodes": [
            {"id": "n1", "graph_id": "article-1", "names": ["Alice"], "kind": "entity"},
            {"id": "n1", "graph_id": "article-1", "names": ["Alice again"], "kind": "entity"},
        ],
        "edges": [],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="duplicate node id"):
        load_graph(path)


def test_load_duplicate_edge_id_raises(tmp_path: Path):
    """Duplicate edge ids are invalid — no silent overwrite."""
    data = {
        "id": "article-1",
        "nodes": [
            {"id": "n1", "graph_id": "article-1", "names": ["Alice"], "kind": "entity"},
            {"id": "v1", "graph_id": "article-1", "names": ["resign"], "kind": "event"},
        ],
        "edges": [
            {"id": "x1", "graph_id": "article-1", "source": "v1", "target": "n1",
             "role": "agent"},
            {"id": "x1", "graph_id": "article-1", "source": "v1", "target": "n1",
             "role": "agent"},
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="duplicate edge id"):
        load_graph(path)


def test_load_unknown_edge_reference_raises(tmp_path: Path):
    """Edge endpoints must resolve to a node — invalid references are
    rejected, never dropped or repaired."""
    data = {
        "id": "article-1",
        "nodes": [
            {"id": "v1", "graph_id": "article-1", "names": ["resign"], "kind": "event"},
        ],
        "edges": [
            {"id": "x1", "graph_id": "article-1", "source": "v1", "target": "n999",
             "role": "agent"},
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unknown node id"):
        load_graph(path)


def test_edge_source_must_be_an_event():
    """Edges express participation: their source is always an event node.
    An entity-sourced edge is invalid state and throws."""
    g = Graph(id="article-1")
    alice = g.add_entity("Alice")
    bob = g.add_entity("Bob")
    g.nodes["v1"] = Node(id="v1", graph_id="article-1", names=["know"], kind="event")
    g.edges["x1"] = Edge(
        id="x1", graph_id="article-1", source=alice.id, target=bob.id, role="agent",
    )

    with pytest.raises(ValueError, match="event"):
        g.validate()

    del g.edges["x1"]
    g.edges["x2"] = Edge(
        id="x2", graph_id="article-1", source="v1", target="v1", role="agent",
    )
    with pytest.raises(ValueError, match="itself"):
        g.validate()


def test_edge_role_must_be_in_vocabulary():
    """A role outside the closed vocabulary is invalid state."""
    g = Graph(id="article-1")
    alice = g.add_entity("Alice")
    g.nodes["v1"] = Node(id="v1", graph_id="article-1", names=["resign"], kind="event")
    g.edges["x1"] = Edge(
        id="x1", graph_id="article-1", source="v1", target=alice.id, role="duration",
    )

    with pytest.raises(ValueError, match="role"):
        g.validate()


def test_load_edge_without_id_raises(tmp_path: Path):
    """Edges lacking an 'id' are invalid — no fallback."""
    data = {
        "id": "article-1",
        "nodes": [
            {"id": "n1", "graph_id": "article-1", "names": ["Alice"], "kind": "entity"},
            {"id": "v1", "graph_id": "article-1", "names": ["know"], "kind": "event"},
        ],
        "edges": [{"graph_id": "article-1", "source": "v1", "target": "n1",
                   "role": "agent"}],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(KeyError):
        load_graph(path)
