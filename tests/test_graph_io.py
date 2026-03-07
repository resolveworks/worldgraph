"""Tests for graph save/load round-trip, including per-node graph_id."""

import json
from pathlib import Path

from worldgraph.graph import Graph, Node, Edge, load_graph, save_graph


def test_save_load_roundtrip_single_graph(tmp_path: Path):
    """Single-article graph: graph_id is omitted per-node and inferred on load."""
    g = Graph(id="article-1")
    n1 = g.add_entity("Alice")
    n2 = g.add_entity("Bob")
    g.add_edge(n1, n2, "knows")

    path = tmp_path / "g.json"
    save_graph(g, path)

    # graph_id should NOT appear in the JSON (all nodes share graph.id)
    with open(path) as f:
        data = json.load(f)
    for node_data in data["nodes"]:
        assert "graph_id" not in node_data

    loaded = load_graph(path)
    for node in loaded.nodes.values():
        assert node.graph_id == "article-1"


def test_save_load_roundtrip_unified_graph(tmp_path: Path):
    """Unified graph with nodes from different source graphs preserves graph_id."""
    g = Graph(id="unified")
    # Manually add nodes with different source graph_ids
    g.nodes["n1"] = Node(id="n1", graph_id="article-1", name="Alice")
    g.nodes["n2"] = Node(id="n2", graph_id="article-2", name="Bob")
    g.nodes["n3"] = Node(id="n3", graph_id="unified", name="Carol")
    g.edges.append(Edge(source="n1", target="n2", relation="knows"))

    path = tmp_path / "unified.json"
    save_graph(g, path)

    # Only nodes with a different graph_id should have it serialized
    with open(path) as f:
        data = json.load(f)
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert nodes_by_id["n1"]["graph_id"] == "article-1"
    assert nodes_by_id["n2"]["graph_id"] == "article-2"
    assert "graph_id" not in nodes_by_id["n3"]

    # Round-trip preserves per-node graph_id
    loaded = load_graph(path)
    assert loaded.nodes["n1"].graph_id == "article-1"
    assert loaded.nodes["n2"].graph_id == "article-2"
    assert loaded.nodes["n3"].graph_id == "unified"
