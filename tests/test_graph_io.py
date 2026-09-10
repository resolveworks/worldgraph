"""Tests for graph save/load round-trip: provenance, multi-label names,
first-class edge identity, and recursive edge references."""

import json
from pathlib import Path

import pytest

from worldgraph.graph import Edge, Graph, Node, load_graph, save_graph


def test_save_load_roundtrip_single_graph(tmp_path: Path):
    """Single-article graph: graph_id is always serialized per node and edge."""
    g = Graph(id="article-1")
    n1 = g.add_entity("Alice")
    n2 = g.add_entity("Bob")
    g.add_edge(n1, n2, "knows")

    path = tmp_path / "g.json"
    save_graph(g, path)

    # graph_id should appear on every node and edge
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


def test_save_load_roundtrip_unified_graph(tmp_path: Path):
    """Unified graph with terms from different source graphs preserves graph_id."""
    g = Graph(id="unified")
    # Manually add terms with different source graph_ids
    g.nodes["n1"] = Node(id="n1", graph_id="article-1", names=["Alice"])
    g.nodes["n2"] = Node(id="n2", graph_id="article-2", names=["Bob"])
    g.nodes["n3"] = Node(id="n3", graph_id="unified", names=["Carol"])
    g.edges["x1"] = Edge(
        id="x1", graph_id="article-1", source="n1", target="n2",
        relation="knows",
    )

    path = tmp_path / "unified.json"
    save_graph(g, path)

    # Every term should have graph_id serialized
    with open(path) as f:
        data = json.load(f)
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert nodes_by_id["n1"]["graph_id"] == "article-1"
    assert nodes_by_id["n2"]["graph_id"] == "article-2"
    assert nodes_by_id["n3"]["graph_id"] == "unified"
    assert data["edges"][0]["graph_id"] == "article-1"

    # Round-trip preserves per-term graph_id
    loaded = load_graph(path)
    assert loaded.nodes["n1"].graph_id == "article-1"
    assert loaded.nodes["n2"].graph_id == "article-2"
    assert loaded.nodes["n3"].graph_id == "unified"
    assert loaded.edges["x1"].graph_id == "article-1"


def test_save_load_roundtrip_multi_label_names(tmp_path: Path):
    """Entities with multiple names survive save/load round-trip."""
    g = Graph(id="article-1")
    n1 = g.add_entity(["Meridian Technologies", "Meridian Tech"])
    n2 = g.add_entity("DataVault")
    g.add_edge(n1, n2, "acquired")

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


def test_save_load_roundtrip_recursive_edges(tmp_path: Path):
    """Edges referencing edges survive save/load round-trip.

    e1: Corin --manage--> Halden Freight
    e2: e1 --role--> managing director
    e3: e1 --for--> Vesterby
    """
    g = Graph(id="article-1")
    corin = g.add_entity("Tessa Corin")
    halden = g.add_entity("Halden Freight")
    role = g.add_entity("managing director")
    vesterby = g.add_entity("Vesterby")
    manage = g.add_edge(corin, halden, "manage")
    g.add_edge(manage, role, "role")
    g.add_edge(manage, vesterby, "for")

    path = tmp_path / "g.json"
    save_graph(g, path)
    loaded = load_graph(path)

    assert len(loaded.nodes) == 4
    assert len(loaded.edges) == 3
    role_edge = next(e for e in g.edges.values() if e.relation == "role")
    assert loaded.edges[role_edge.id].source == manage.id
    assert loaded.resolve(role_edge.source) is loaded.edges[manage.id]

    # the 'for' edge targets the Vesterby entity node
    for_edge = next(e for e in g.edges.values() if e.relation == "for")
    assert loaded.resolve(loaded.edges[for_edge.id].target) is loaded.nodes[vesterby.id]


def test_forward_edge_references(tmp_path: Path):
    """An edge may reference an edge that appears later in the JSON —
    construction order carries no semantics."""
    data = {
        "id": "article-1",
        "nodes": [
            {"id": "n1", "graph_id": "article-1", "names": ["Alice"]},
            {"id": "n2", "graph_id": "article-1", "names": ["Bob"]},
        ],
        "edges": [
            # x1 references x2, defined after it
            {"id": "x1", "graph_id": "article-1", "source": "x2",
             "target": "n1", "relation": "denies"},
            {"id": "x2", "graph_id": "article-1", "source": "n1",
             "target": "n2", "relation": "knows"},
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    loaded = load_graph(path)
    assert loaded.edges["x1"].source == "x2"
    assert loaded.resolve("x1").source == "x2"


def test_load_duplicate_node_id_raises(tmp_path: Path):
    """Duplicate node ids are invalid — no silent overwrite."""
    data = {
        "id": "article-1",
        "nodes": [
            {"id": "n1", "graph_id": "article-1", "names": ["Alice"]},
            {"id": "n1", "graph_id": "article-1", "names": ["Alice again"]},
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
        "nodes": [{"id": "n1", "graph_id": "article-1", "names": ["Alice"]}],
        "edges": [
            {"id": "x1", "graph_id": "article-1", "source": "n1",
             "target": "n1", "relation": "knows"},
            {"id": "x1", "graph_id": "article-1", "source": "n1",
             "target": "n1", "relation": "knows"},
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="duplicate edge id"):
        load_graph(path)


def test_load_unknown_edge_reference_raises(tmp_path: Path):
    """Edge endpoints must resolve to a node or edge — invalid references
    are rejected, never dropped or repaired."""
    data = {
        "id": "article-1",
        "nodes": [{"id": "n1", "graph_id": "article-1", "names": ["Alice"]}],
        "edges": [
            {"id": "x1", "graph_id": "article-1", "source": "n1",
             "target": "n999", "relation": "knows"},
        ],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="references unknown term id"):
        load_graph(path)


def test_node_and_edge_ids_must_be_disjoint():
    """Nodes and edges share one reference namespace — a colliding id would
    make references ambiguous and is invalid."""
    g = Graph(id="article-1")
    g.nodes["t1"] = Node(id="t1", graph_id="article-1", names=["Alice"])
    g.edges["t1"] = Edge(
        id="t1", graph_id="article-1", source="t1", target="t1",
        relation="knows",
    )

    with pytest.raises(ValueError, match="disjoint"):
        g.validate()


def test_load_edge_without_id_raises(tmp_path: Path):
    """Edges lacking an 'id' are invalid — no fallback."""
    data = {
        "id": "article-1",
        "nodes": [{"id": "n1", "graph_id": "article-1", "names": ["Alice"]}],
        "edges": [{"graph_id": "article-1", "source": "n1", "target": "n1",
                   "relation": "knows"}],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(KeyError):
        load_graph(path)
