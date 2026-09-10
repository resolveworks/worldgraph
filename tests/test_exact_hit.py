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
    return Extraction(
        entities=[Entity(id=f"e{i+1}", name=n) for i, n in enumerate(entities)],
        relations=[
            Relation(source=f"e{s}", target=f"e{t}", relation=r) for s, t, r in relations
        ],
    )


GOLD = extraction(
    ["Acme Corp", "Gamma AI"],
    [(1, 2, "acquired")],
)


def test_exact_hit_despite_reordered_entities_and_different_ids():
    pred = Extraction(
        entities=[Entity(id="z9", name="Gamma AI"), Entity(id="z8", name="Acme Corp")],
        relations=[Relation(source="z8", target="z9", relation="acquired")],
    )
    assert hit(pred, GOLD)


def test_missing_entity_is_a_miss():
    assert not hit(extraction(["Acme Corp"], []), GOLD)


def test_extra_entity_is_a_miss():
    assert not hit(extraction(["Acme Corp", "Gamma AI", "Delta Labs"], [(1, 2, "acquired")]), GOLD)


def test_paraphrased_relation_is_a_miss():
    assert not hit(extraction(["Acme Corp", "Gamma AI"], [(1, 2, "purchased")]), GOLD)


def test_extra_edge_is_a_miss():
    pred = extraction(["Acme Corp", "Gamma AI"], [(1, 2, "acquired"), (2, 2, "acquired")])
    assert not hit(pred, GOLD)


def test_duplicate_entity_is_a_miss():
    pred = extraction(["Acme Corp", "Gamma AI", "Gamma AI"], [(1, 2, "acquired")])
    assert not hit(pred, GOLD)
