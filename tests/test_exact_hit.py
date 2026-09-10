"""ExactHit evaluator behavior on hand-constructed extraction pairs.

Pure comparison — no embeddings, no LLM calls. One assumption per test.
"""

from pydantic_evals.evaluators import EvaluatorContext

from evals.run import ExactHit
from worldgraph.extract import Entity, Extraction, Relation


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


def extraction(entities, relations) -> Extraction:
    """Entities by name (ids e1..); relations as (source, target, relation,
    temporal) with ids r1.. — source and target may reference either."""
    return Extraction(
        entities=[Entity(id=f"e{i + 1}", name=n) for i, n in enumerate(entities)],
        relations=[
            Relation(id=f"r{i + 1}", source=s, target=t, relation=r, temporal=tmp)
            for i, (s, t, r, tmp) in enumerate(relations)
        ],
    )


GOLD = extraction(
    ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
    [
        ("e1", "e2", "manage", "current"),
        ("r1", "e3", "for", "current"),
        ("r1", "e4", "role", "current"),
    ],
)


def test_hit_despite_reordered_terms_and_different_ids():
    """Qualifiers listed before the relation they reference, entities and
    relations in a different order, all ids renamed — still an exact hit."""
    pred = Extraction(
        entities=[
            Entity(id="q4", name="managing director"),
            Entity(id="q2", name="Halden Freight"),
            Entity(id="q1", name="Tessa Corin"),
            Entity(id="q3", name="Vesterby"),
        ],
        relations=[
            Relation(
                id="s2", source="s1", target="q3", relation="for", temporal="current"
            ),
            Relation(
                id="s3", source="s1", target="q4", relation="role", temporal="current"
            ),
            Relation(
                id="s1", source="q1", target="q2", relation="manage", temporal="current"
            ),
        ],
    )
    assert hit(pred, GOLD)


def test_missing_qualifier_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [("e1", "e2", "manage", "current"), ("r1", "e3", "for", "current")],
    )
    assert not hit(pred, GOLD)


def test_wrong_qualifier_attachment_is_a_miss():
    """Same entities, same phrases, same temporals — but the role and scope
    qualifiers are attached to each other's values."""
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [
            ("e1", "e2", "manage", "current"),
            ("r1", "e4", "for", "current"),
            ("r1", "e3", "role", "current"),
        ],
    )
    assert not hit(pred, GOLD)


def test_qualifier_on_wrong_relation_occurrence_is_a_miss():
    """Two relations share the phrase 'manage'; the qualifier must land on
    the past occurrence, not the current one — the phrase alone cannot
    distinguish them."""
    entities = ["Tessa Corin", "Halden Freight", "Meridian Rail", "operations director"]
    gold = extraction(
        entities,
        [
            ("e1", "e2", "manage", "current"),
            ("e1", "e3", "manage", "past"),
            ("r2", "e4", "role", "past"),
        ],
    )
    pred = extraction(
        entities,
        [
            ("e1", "e2", "manage", "current"),
            ("e1", "e3", "manage", "past"),
            ("r1", "e4", "role", "past"),
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
        [
            ("e1", "e2", "manage", "current"),
            ("r1", "e3", "for", "current"),
            ("r1", "e4", "role", "current"),
        ],
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
        [
            ("e1", "e3", "manage", "current"),
            ("r1", "e4", "for", "current"),
            ("r1", "e5", "role", "current"),
        ],
    )
    assert not hit(pred, GOLD)


def test_extra_edge_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [
            ("e1", "e2", "manage", "current"),
            ("r1", "e3", "for", "current"),
            ("r1", "e4", "role", "current"),
            ("e2", "e3", "be based in", "current"),
        ],
    )
    assert not hit(pred, GOLD)


def test_changed_phrase_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [
            ("e1", "e2", "run", "current"),
            ("r1", "e3", "for", "current"),
            ("r1", "e4", "role", "current"),
        ],
    )
    assert not hit(pred, GOLD)


def test_changed_temporal_is_a_miss():
    pred = extraction(
        ["Tessa Corin", "Halden Freight", "Vesterby", "managing director"],
        [
            ("e1", "e2", "manage", "past"),
            ("r1", "e3", "for", "current"),
            ("r1", "e4", "role", "current"),
        ],
    )
    assert not hit(pred, GOLD)
