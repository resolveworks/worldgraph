"""Shared graph data structures and I/O."""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from worldgraph.constants import NAME_EDGE


@dataclass
class Node:
    id: str
    graph_id: str


@dataclass
class LiteralNode(Node):
    label: str = ""


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

    def add_entity(self, name: str) -> Node:
        """Add an entity node with an "is named" edge to a literal node."""
        entity = Node(id=str(uuid.uuid4()), graph_id=self.id)
        literal = LiteralNode(id=str(uuid.uuid4()), graph_id=self.id, label=name)
        self.nodes[entity.id] = entity
        self.nodes[literal.id] = literal
        self.edges.append(Edge(source=entity.id, target=literal.id, relation=NAME_EDGE))
        return entity

    def add_edge(self, source: Node, target: Node, relation: str) -> None:
        """Add a relation edge between two existing nodes."""
        self.edges.append(Edge(source=source.id, target=target.id, relation=relation))


def load_graphs(graphs_dir: Path) -> list[Graph]:
    """Load per-article graph JSON files from a directory.

    Each graph's id is the article_id; each node's graph_id tracks its origin.
    """
    graphs: list[Graph] = []

    for path in sorted(graphs_dir.glob("*.json")):
        with open(path) as f:
            g = json.load(f)

        graph_id = g["id"]
        nodes: dict[str, Node] = {}

        for n in g["nodes"]:
            nid = n["id"]
            if "label" in n:
                nodes[nid] = LiteralNode(id=nid, graph_id=graph_id, label=n["label"])
            else:
                nodes[nid] = Node(id=nid, graph_id=graph_id)

        edges: list[Edge] = []
        for ed in g["edges"]:
            edges.append(
                Edge(source=ed["source"], target=ed["target"], relation=ed["relation"])
            )

        graphs.append(Graph(id=graph_id, nodes=nodes, edges=edges))

    return graphs


def entity_names(graph: Graph, eid: str) -> list[str]:
    """Get the names of an entity by following its NAME_EDGE edges."""
    names = []
    for edge in graph.edges:
        if edge.relation == NAME_EDGE and edge.source == eid:
            tgt = graph.nodes.get(edge.target)
            if isinstance(tgt, LiteralNode):
                names.append(tgt.label)
    return names if names else [eid]


def save_graph(
    graph: Graph,
    path: Path,
    matches: list[list[str]] | None = None,
) -> None:
    """Write graph to JSON, with optional match groups."""
    nodes_out = []
    for n in graph.nodes.values():
        if isinstance(n, LiteralNode):
            nodes_out.append({"id": n.id, "label": n.label})
        else:
            nodes_out.append({"id": n.id})

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
