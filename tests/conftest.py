from dotenv import load_dotenv

from worldgraph.graph import Graph, Node

load_dotenv()


def fact(
    graph: Graph,
    label: str,
    agent: Node | None = None,
    patients: tuple[Node, ...] = (),
) -> Node:
    """Add an event node with role edges to its participants.

    Events are the facts of a graph: an event node (its label is never used
    for matching) connected by role edges to the entities or events that
    participate in it.
    """
    event = graph.add_event(label)
    if agent is not None:
        graph.add_edge(event, agent, "agent")
    for patient in patients:
        graph.add_edge(event, patient, "patient")
    return event
