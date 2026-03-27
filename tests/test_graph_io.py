"""Tests for graph save/load round-trip, including per-node graph_id and multi-label names."""

import json
from pathlib import Path

from worldgraph.graph import Graph, Node, Edge, load_graph, save_graph


def test_save_load_roundtrip_single_graph(tmp_path: Path):
    """Single-article graph: graph_id is always serialized per node."""
    g = Graph(id="article-1")
    n1 = g.add_entity("Alice")
    n2 = g.add_entity("Bob")
    g.add_edge(n1, n2, "knows")

    path = tmp_path / "g.json"
    save_graph(g, path)

    # graph_id should appear on every node
    with open(path) as f:
        data = json.load(f)
    for node_data in data["nodes"]:
        assert node_data["graph_id"] == "article-1"

    loaded = load_graph(path)
    for node in loaded.nodes.values():
        assert node.graph_id == "article-1"


def test_save_load_roundtrip_unified_graph(tmp_path: Path):
    """Unified graph with nodes from different source graphs preserves graph_id."""
    g = Graph(id="unified")
    # Manually add nodes with different source graph_ids
    g.nodes["n1"] = Node(id="n1", graph_id="article-1", names=["Alice"])
    g.nodes["n2"] = Node(id="n2", graph_id="article-2", names=["Bob"])
    g.nodes["n3"] = Node(id="n3", graph_id="unified", names=["Carol"])
    g.edges.append(Edge(source="n1", target="n2", relation="knows"))

    path = tmp_path / "unified.json"
    save_graph(g, path)

    # Every node should have graph_id serialized
    with open(path) as f:
        data = json.load(f)
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert nodes_by_id["n1"]["graph_id"] == "article-1"
    assert nodes_by_id["n2"]["graph_id"] == "article-2"
    assert nodes_by_id["n3"]["graph_id"] == "unified"

    # Round-trip preserves per-node graph_id
    loaded = load_graph(path)
    assert loaded.nodes["n1"].graph_id == "article-1"
    assert loaded.nodes["n2"].graph_id == "article-2"
    assert loaded.nodes["n3"].graph_id == "unified"


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


def test_load_legacy_single_name_format(tmp_path: Path):
    """Loading a graph saved with the old single-name format works."""
    data = {
        "id": "legacy",
        "nodes": [
            {"id": "n1", "graph_id": "legacy", "name": "Alice"},
            {"id": "n2", "graph_id": "legacy", "name": "Bob"},
        ],
        "edges": [{"source": "n1", "target": "n2", "relation": "knows"}],
        "matches": [],
    }
    path = tmp_path / "legacy.json"
    with open(path, "w") as f:
        json.dump(data, f)

    loaded = load_graph(path)
    assert loaded.nodes["n1"].names == ["Alice"]
    assert loaded.nodes["n2"].names == ["Bob"]
