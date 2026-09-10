"""ExactHit evaluator behavior on hand-constructed extraction pairs.

Pure comparison — no LLM calls. One assumption per test.
"""

from pydantic_evals.evaluators import EvaluatorContext

from evals.run import ExactHit
from worldgraph.extract import Entity, Event, Extraction, Participant


def hit(output: Extraction, golden: Extraction) -> bool:
    ctx = EvaluatorContext(
        name="test",
        inputs=None,
        metadata=None,
        expected_output=golden,
        output=output,
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )
    return ExactHit().evaluate(ctx)


def extraction(entities, events) -> Extraction:
    """Entities by name (ids e1..); events as (label, [(role, ref), ...])
    with ids v1.. — participant refs may name entities or events."""
    return Extraction(
        entities=[Entity(id=f"e{i + 1}", name=n) for i, n in enumerate(entities)],
        events=[
            Event(
                id=f"v{i + 1}",
                label=label,
                participants=[Participant(role=role, ref=ref) for role, ref in participants],
            )
            for i, (label, participants) in enumerate(events)
        ],
    )


GOLD = extraction(
    ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
    [
        ("manage", [("agent", "e1"), ("patient", "e2"), ("patient", "e3"), ("patient", "e4")]),
    ],
)


def test_hit_despite_reordered_terms_and_different_ids():
    """Participants listed in a different order, entities and events in a
    different order, all ids renamed — still an exact hit."""
    pred = Extraction(
        entities=[
            Entity(id="q4", name="managing director"),
            Entity(id="q2", name="Halden Freight"),
            Entity(id="q1", name="Tessa Corin"),
            Entity(id="q3", name="Vesterby"),
        ],
        events=[
            Event(
                id="s1",
                label="manage",
                participants=[
                    Participant(role="patient", ref="q4"),
                    Participant(role="patient", ref="q3"),
                    Participant(role="patient", ref="q2"),
                    Participant(role="agent", ref="q1"),
                ],
            ),
        ],
    )
    assert hit(pred, GOLD)


def test_missing_participant_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [("manage", [("agent", "e1"), ("patient", "e2"), ("patient", "e3")])],
    )
    assert not hit(pred, GOLD)


def test_swapped_roles_are_a_miss():
    """Same entities, same label — but agent and patient swapped. Roles
    are the only structure the matcher sees; getting them wrong is wrong."""
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [("manage", [("agent", "e2"), ("patient", "e1"), ("patient", "e3"), ("patient", "e4")])],
    )
    assert not hit(pred, GOLD)


def test_participant_on_wrong_event_occurrence_is_a_miss():
    """Two events share the label 'manage'; the qualifier must land on
    the Meridian Rail occurrence, not the Halden Freight one — the label
    alone cannot distinguish them."""
    entities = ["Tessa Corin", "Halden Freight", "Meridian Rail", "operations director"]
    gold = extraction(
        entities,
        [
            ("manage", [("agent", "e1"), ("patient", "e2")]),
            ("manage", [("agent", "e1"), ("patient", "e3"), ("patient", "e4")]),
        ],
    )
    pred = extraction(
        entities,
        [
            ("manage", [("agent", "e1"), ("patient", "e2"), ("patient", "e4")]),
            ("manage", [("agent", "e1"), ("patient", "e3")]),
        ],
    )
    assert not hit(pred, gold)


def test_nested_event_reference_targets_the_right_occurrence():
    """Ivo joins the visit to Kalden, not the visit to Vesterby: the
    nested event form must resolve to the specific occurrence."""
    entities = ["Marisol Vaneck", "Vesterby", "Kalden", "Ivo Brandt"]
    gold = extraction(
        entities,
        [
            ("visit", [("agent", "e1"), ("patient", "e2")]),
            ("visit", [("agent", "e1"), ("patient", "e3")]),
            ("join", [("agent", "e4"), ("patient", "v2")]),
        ],
    )
    pred = extraction(
        entities,
        [
            ("visit", [("agent", "e1"), ("patient", "e2")]),
            ("visit", [("agent", "e1"), ("patient", "e3")]),
            ("join", [("agent", "e4"), ("patient", "v1")]),
        ],
    )
    assert not hit(pred, gold)


def test_missing_entity_is_a_miss():
    assert not hit(extraction(["Tessa Corin"], []), GOLD)


def test_extra_entity_is_a_miss():
    pred = extraction(
        [
            "Tessa Corin",
            "Halden Freight",
            "Vesterby",
            "managing director",
            "Meridian Rail",
        ],
        [("manage", [("agent", "e1"), ("patient", "e2"), ("patient", "e3"), ("patient", "e4")])],
    )
    assert not hit(pred, GOLD)


def test_duplicate_entity_name_is_a_miss():
    pred = extraction(
        [
            "Tessa Corin",
            "Tessa Corin",
            "Halden Freight",
            "Vesterby",
            "managing director",
        ],
        [("manage", [("agent", "e1"), ("patient", "e3"), ("patient", "e4"), ("patient", "e5")])],
    )
    assert not hit(pred, GOLD)


def test_extra_event_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [
            ("manage", [("agent", "e1"), ("patient", "e2"), ("patient", "e3"), ("patient", "e4")]),
            ("be based in", [("agent", "e2"), ("patient", "e3")]),
        ],
    )
    assert not hit(pred, GOLD)


def test_changed_label_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [("run", [("agent", "e1"), ("patient", "e2"), ("patient", "e3"), ("patient", "e4")])],
    )
    assert not hit(pred, GOLD)
