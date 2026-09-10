"""Shared graph data structures and I/O.

A graph is bipartite: **entity** nodes (things in the world) and **event**
nodes (facts asserted by the article), connected by participation edges.
An edge's source is always an event node; its target is a participant —
an entity or another event (joining a visit, causing a suspension). The
edge label is a role from a closed vocabulary: the matcher aligns
participants by exact role equality, so the vocabulary is enforced here,
at the data-structure boundary.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

NodeKind = Literal["entity", "event"]

Role = Literal["agent", "patient"]
ROLES: frozenset[str] = frozenset({"agent", "patient"})


@dataclass
class Node:
    id: str
    graph_id: str
    names: list[str]
    kind: NodeKind


@dataclass
class Edge:
    """A participation: an event node connected to one of its participants.

    ``source`` is the event node id, ``target`` the participant node id
    (an entity or another event), ``role`` the participant's role.
    """

    id: str
    graph_id: str  # id of the article graph this edge was extracted from
    source: str  # Node.id of kind "event"
    target: str  # Node.id of any kind
    role: str


@dataclass
class Graph:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def add_entity(self, names: str | list[str]) -> Node:
        """Add an entity node with the given name(s)."""
        if isinstance(names, str):
            names = [names]
        entity = Node(id=str(uuid.uuid4()), graph_id=self.id, names=names, kind="entity")
        self.nodes[entity.id] = entity
        return entity

    def add_event(self, label: str) -> Node:
        """Add an event node. The label names the node for output and
        display; it is never used for matching."""
        event = Node(id=str(uuid.uuid4()), graph_id=self.id, names=[label], kind="event")
        self.nodes[event.id] = event
        return event

    def add_edge(
        self,
        source: Node | str,
        target: Node | str,
        role: Role,
        id: str | None = None,
    ) -> Edge:
        """Add a participation edge from an event node to a participant.

        Endpoints may be given as node objects or as ids.  Ids (and the
        optional explicit edge ``id``) exist for construction flexibility;
        ``validate()`` checks all invariants once construction is complete.
        """
        src = source.id if isinstance(source, Node) else source
        tgt = target.id if isinstance(target, Node) else target
        if isinstance(source, Node) and source.kind != "event":
            raise ValueError(
                f"edge source must be an event node, got kind {source.kind!r}"
            )
        edge = Edge(
            id=id if id is not None else str(uuid.uuid4()),
            graph_id=self.id,
            source=src,
            target=tgt,
            role=role,
        )
        self.edges[edge.id] = edge
        return edge

    def validate(self) -> None:
        """Check the structural invariants.

        Every edge endpoint must resolve to a node of this graph, the edge
        source must be an event node, no event may participate in itself,
        and the role must be in the closed vocabulary.
        """
        for edge in self.edges.values():
            for endpoint in (edge.source, edge.target):
                if endpoint not in self.nodes:
                    raise ValueError(
                        f"edge {edge.id!r} references unknown node id: {endpoint!r}"
                    )
            if self.nodes[edge.source].kind != "event":
                raise ValueError(
                    f"edge {edge.id!r} source is not an event node: {edge.source!r}"
                )
            if edge.source == edge.target:
                raise ValueError(f"event participates in itself: {edge.source!r}")
            if edge.role not in ROLES:
                raise ValueError(f"edge {edge.id!r} has unknown role: {edge.role!r}")


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
            kind=node_data["kind"],
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
            role=edge_data["role"],
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
            {
                "id": node.id,
                "graph_id": node.graph_id,
                "names": node.names,
                "kind": node.kind,
            }
        )

    edges_out = []
    for edge in graph.edges.values():
        edges_out.append(
            {
                "id": edge.id,
                "graph_id": edge.graph_id,
                "source": edge.source,
                "target": edge.target,
                "role": edge.role,
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
