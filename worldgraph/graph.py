"""Shared graph data structures and I/O."""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    id: str
    graph_id: str
    names: list[str]


@dataclass
class Edge:
    source: str  # node id
    target: str  # node id
    relation: str


@dataclass
class Graph:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_entity(self, names: str | list[str]) -> Node:
        """Add an entity node with the given name(s)."""
        if isinstance(names, str):
            names = [names]
        entity = Node(id=str(uuid.uuid4()), graph_id=self.id, names=names)
        self.nodes[entity.id] = entity
        return entity

    def add_edge(self, source: Node, target: Node, relation: str) -> None:
        """Add a relation edge between two existing nodes."""
        self.edges.append(Edge(source=source.id, target=target.id, relation=relation))


def load_graph(path: Path) -> Graph:
    """Load a single graph JSON file."""
    with open(path) as f:
        data = json.load(f)

    graph_id = data["id"]
    nodes: dict[str, Node] = {}

    for node_data in data["nodes"]:
        node_id = node_data["id"]
        raw_names = node_data.get("names") or [node_data["name"]]
        nodes[node_id] = Node(
            id=node_id,
            graph_id=node_data["graph_id"],
            names=raw_names if isinstance(raw_names, list) else [raw_names],
        )

    edges: list[Edge] = []
    for edge_data in data["edges"]:
        edges.append(
            Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                relation=edge_data["relation"],
            )
        )

    return Graph(id=graph_id, nodes=nodes, edges=edges)


def save_graph(
    graph: Graph,
    path: Path,
    matches: list[list[str]] | None = None,
) -> None:
    """Write graph to JSON, with optional match groups."""
    nodes_out = []
    for node in graph.nodes.values():
        nodes_out.append(
            {"id": node.id, "graph_id": node.graph_id, "names": node.names}
        )

    edges_out = []
    for edge in graph.edges:
        edges_out.append(
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
            }
        )

    output = {
        "id": graph.id,
        "nodes": nodes_out,
        "edges": edges_out,
        "matches": matches or [],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
