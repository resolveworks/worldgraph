from dotenv import load_dotenv

from worldgraph.graph import ROLES, Graph, Node

load_dotenv()


def fact(
    graph: Graph,
    label: str,
    **roles: Node | tuple[Node, ...],
) -> Node:
    """Add an event node with role edges to its participants.

    Events are the facts of a graph: an event node (its label is never used
    for matching) connected by role edges to the entities or events that
    participate in it. Each kwarg is one role from the closed vocabulary
    (worldgraph.graph.ROLES) with one participant or a tuple of them:

        fact(g, "acquire", agent=apple, patient=beats)
        fact(g, "employ", agent=elena, patient=meridian, capacity=ceo_title)
    """
    unknown = sorted(set(roles) - ROLES)
    if unknown:
        raise ValueError(f"unknown participant roles: {unknown}")
    event = graph.add_event(label)
    for role, participant in roles.items():
        participants = participant if isinstance(participant, tuple) else (participant,)
        for node in participants:
            graph.add_edge(event, node, role)
    return event
