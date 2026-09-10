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
    """A relation occurrence — a first-class graph term.

    ``source``/``target`` reference a node id or another edge id, so
    qualifiers of a fact (role, scope, attribution, modality, nested
    claims) are themselves edges attached to the edge they qualify.
    """

    id: str
    graph_id: str  # id of the article graph this edge was extracted from
    source: str  # Node.id or Edge.id
    target: str  # Node.id or Edge.id
    relation: str


# Nodes and edges are addressable graph terms sharing one id namespace.
Term = Node | Edge


@dataclass
class Graph:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def add_entity(self, names: str | list[str]) -> Node:
        """Add an entity node with the given name(s)."""
        if isinstance(names, str):
            names = [names]
        entity = Node(id=str(uuid.uuid4()), graph_id=self.id, names=names)
        self.nodes[entity.id] = entity
        return entity

    def add_edge(
        self,
        source: Term | str,
        target: Term | str,
        relation: str,
        id: str | None = None,
    ) -> Edge:
        """Add a relation edge between two terms — nodes or edges.

        Endpoints may be given as term objects or as ids.  Ids (and the
        optional explicit edge ``id``) exist for forward references: an
        edge can reference an edge that is added later.  Call
        ``validate()`` once construction is complete.
        """
        src = source.id if isinstance(source, (Node, Edge)) else source
        tgt = target.id if isinstance(target, (Node, Edge)) else target
        edge = Edge(
            id=id if id is not None else str(uuid.uuid4()),
            graph_id=self.id,
            source=src,
            target=tgt,
            relation=relation,
        )
        self.edges[edge.id] = edge
        return edge

    def resolve(self, term_id: str) -> Term:
        """Look up a graph term by id — a node or an edge."""
        if term_id in self.nodes:
            return self.nodes[term_id]
        if term_id in self.edges:
            return self.edges[term_id]
        raise ValueError(f"unknown term id: {term_id!r}")

    def validate(self) -> None:
        """Check the structural invariants.

        Node and edge ids must be disjoint (they share one reference
        namespace), and every edge endpoint must resolve to a term of
        this graph. Nesting depth is not limited.
        """
        shared = sorted(self.nodes.keys() & self.edges.keys())
        if shared:
            raise ValueError(f"node and edge ids must be disjoint, shared: {shared}")
        for edge in self.edges.values():
            for role, endpoint in (("source", edge.source), ("target", edge.target)):
                if endpoint not in self.nodes and endpoint not in self.edges:
                    raise ValueError(
                        f"edge {edge.id!r} {role} references unknown term id: "
                        f"{endpoint!r}"
                    )


def load_graph(path: Path) -> Graph:
    """Load a single graph JSON file.

    Raises on duplicate node/edge ids and on edge references that do not
    resolve — invalid state is never silently repaired.
    """
    with open(path) as f:
        data = json.load(f)

    graph_id = data["id"]
    nodes: dict[str, Node] = {}
    for node_data in data["nodes"]:
        node_id = node_data["id"]
        if node_id in nodes:
            raise ValueError(f"duplicate node id: {node_id!r}")
        nodes[node_id] = Node(
            id=node_id,
            graph_id=node_data["graph_id"],
            names=node_data["names"],
        )

    edges: dict[str, Edge] = {}
    for edge_data in data["edges"]:
        edge_id = edge_data["id"]
        if edge_id in edges:
            raise ValueError(f"duplicate edge id: {edge_id!r}")
        edges[edge_id] = Edge(
            id=edge_id,
            graph_id=edge_data["graph_id"],
            source=edge_data["source"],
            target=edge_data["target"],
            relation=edge_data["relation"],
        )

    graph = Graph(id=graph_id, nodes=nodes, edges=edges)
    graph.validate()
    return graph


def save_graph(
    graph: Graph,
    path: Path,
    matches: list[list[str]] | None = None,
) -> None:
    """Write graph to JSON, with optional match groups. Validates first."""
    graph.validate()

    nodes_out = []
    for node in graph.nodes.values():
        nodes_out.append(
            {"id": node.id, "graph_id": node.graph_id, "names": node.names}
        )

    edges_out = []
    for edge in graph.edges.values():
        edges_out.append(
            {
                "id": edge.id,
                "graph_id": edge.graph_id,
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
