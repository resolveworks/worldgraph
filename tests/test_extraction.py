"""Tests for the extraction schema: unique ids, valid participant
references, the closed role vocabulary, and conversion to a runtime graph.
Event participants may reference entities or events, including forward
references between events."""

import pytest
from pydantic import ValidationError

from worldgraph.extract import (
    Entity,
    Event,
    Extraction,
    Participant,
    extraction_to_graph,
)


def entities(*names: str) -> list[Entity]:
    return [Entity(id=f"e{i + 1}", name=name) for i, name in enumerate(names)]


def test_duplicate_entity_id_raises():
    with pytest.raises(ValidationError, match="duplicate entity ids"):
        Extraction(
            entities=[Entity(id="e1", name="Alice"), Entity(id="e1", name="Bob")],
            events=[],
        )


def test_duplicate_event_id_raises():
    with pytest.raises(ValidationError, match="duplicate event ids"):
        Extraction(
            entities=entities("Alice", "Bob"),
            events=[
                Event(
                    id="v1",
                    label="know",
                    participants=[Participant(role="agent", ref="e1")],
                ),
                Event(
                    id="v1",
                    label="meet",
                    participants=[Participant(role="agent", ref="e2")],
                ),
            ],
        )


def test_entity_event_id_collision_raises():
    """Entities and events share one reference namespace — a collision
    would make participant references ambiguous."""
    with pytest.raises(ValidationError, match="disjoint"):
        Extraction(
            entities=entities("Alice"),
            events=[
                Event(
                    id="e1",
                    label="resign",
                    participants=[Participant(role="agent", ref="e1")],
                ),
            ],
        )


def test_unknown_participant_reference_raises():
    """References that resolve to neither an entity nor an event are
    rejected — never dropped or patched."""
    with pytest.raises(ValidationError, match="unknown ids"):
        Extraction(
            entities=entities("Alice"),
            events=[
                Event(
                    id="v1",
                    label="resign",
                    participants=[Participant(role="agent", ref="e99")],
                ),
            ],
        )


def test_event_cannot_participate_in_itself():
    with pytest.raises(ValidationError, match="itself"):
        Extraction(
            entities=entities("Alice"),
            events=[
                Event(
                    id="v1",
                    label="cause",
                    participants=[Participant(role="agent", ref="v1")],
                ),
            ],
        )


def test_event_without_participants_raises():
    with pytest.raises(ValidationError):
        Extraction(entities=entities("Alice"), events=[Event(id="v1", label="resign", participants=[])])


def test_role_vocabulary_is_closed():
    """Roles outside the closed nine-role vocabulary are rejected at the
    schema boundary — the matcher aligns participants by exact role
    equality, so role consistency is enforced structurally, not by prompt
    discipline."""
    with pytest.raises(ValidationError):
        Participant(role="co-agent", ref="e1")


def test_event_participant_may_reference_another_event():
    """Events participate in other events: joining a visit, causing a
    suspension. Forward references between events are valid."""
    ext = Extraction(
        entities=entities("Ivo Brandt"),
        events=[
            Event(
                id="v2",
                label="join",
                participants=[
                    Participant(role="agent", ref="e1"),
                    Participant(role="patient", ref="v1"),
                ],
            ),
            Event(
                id="v1",
                label="visit",
                participants=[Participant(role="agent", ref="e1")],
            ),
        ],
    )
    assert ext.events[0].participants[1].ref == "v1"


# ---------------------------------------------------------------------------
# Conversion to runtime graph
# ---------------------------------------------------------------------------


def qualifier_extraction() -> Extraction:
    """The Corin example: one manage event whose qualifiers (title, scope)
    are participants with their proper roles — capacity and beneficiary."""
    return Extraction(
        entities=entities("Tessa Corin", "Halden Freight", "Vesterby", "managing director"),
        events=[
            Event(
                id="v1",
                label="manage",
                participants=[
                    Participant(role="agent", ref="e1"),
                    Participant(role="patient", ref="e2"),
                    Participant(role="capacity", ref="e4"),
                    Participant(role="beneficiary", ref="e3"),
                ],
            ),
        ],
    )


def test_extraction_to_graph_kinds_and_roles():
    """Entities become entity nodes, events become event nodes named by
    their label, and participants become role edges from the event."""
    graph = extraction_to_graph("article-1", qualifier_extraction())

    entities = [n for n in graph.nodes.values() if n.kind == "entity"]
    events = [n for n in graph.nodes.values() if n.kind == "event"]
    assert len(entities) == 4
    assert len(events) == 1
    assert events[0].names == ["manage"]

    roles = sorted(edge.role for edge in graph.edges.values())
    assert roles == ["agent", "beneficiary", "capacity", "patient"]
    assert all(edge.source == events[0].id for edge in graph.edges.values())


def test_extraction_to_graph_event_references():
    """An event participating in another event produces an edge between
    the two event nodes."""
    ext = Extraction(
        entities=entities("Ivo Brandt"),
        events=[
            Event(
                id="v1",
                label="visit",
                participants=[Participant(role="agent", ref="e1")],
            ),
            Event(
                id="v2",
                label="join",
                participants=[
                    Participant(role="agent", ref="e1"),
                    Participant(role="patient", ref="v1"),
                ],
            ),
        ],
    )
    graph = extraction_to_graph("article-1", ext)

    visit = next(n for n in graph.nodes.values() if n.names == ["visit"])
    join = next(n for n in graph.nodes.values() if n.names == ["join"])
    join_patient = next(
        e for e in graph.edges.values() if e.source == join.id and e.role == "patient"
    )
    assert join_patient.target == visit.id


def test_extraction_to_graph_provenance():
    """Every node and edge carries the article graph id as provenance."""
    graph = extraction_to_graph("article-1", qualifier_extraction())

    assert all(node.graph_id == "article-1" for node in graph.nodes.values())
    assert all(edge.graph_id == "article-1" for edge in graph.edges.values())


def test_extraction_to_graph_unique_node_ids():
    """Entities and events map to distinct runtime ids — extraction-local
    ids ('e1', 'v1') never collide at runtime."""
    graph = extraction_to_graph("article-1", qualifier_extraction())

    assert len(graph.nodes) == 5  # 4 entities + 1 event
