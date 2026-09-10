"""ExtractionPRF evaluator behavior on hand-constructed extraction pairs.

Real embeddings via the session-scoped ``embedder`` fixture — no DeepSeek
calls. One assumption per test.
"""

import pytest
from pydantic_evals.evaluators import EvaluatorContext

from evals.run import ExtractionPRF
from worldgraph.extract import Entity, Extraction, Relation


def score(pred: Extraction, gold: Extraction, embedder) -> dict[str, float]:
    ctx = EvaluatorContext(
        name="test",
        inputs=None,
        metadata=None,
        expected_output=gold,
        output=pred,
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )
    return ExtractionPRF(embedder).evaluate(ctx)


def extraction(entities, relations):
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


def test_perfect_match_scores_one(embedder):
    assert score(GOLD, GOLD, embedder) == {
        "entity_precision": 1.0,
        "entity_recall": 1.0,
        "entity_f1": 1.0,
        "edge_precision": 1.0,
        "edge_recall": 1.0,
        "edge_f1": 1.0,
    }


def test_missed_golden_entity_drops_recall(embedder):
    pred = extraction(["Acme Corp"], [])
    s = score(pred, GOLD, embedder)
    assert s["entity_recall"] == 0.5
    assert s["entity_precision"] == 1.0


def test_extra_predicted_entity_drops_precision(embedder):
    pred = extraction(["Acme Corp", "Gamma AI", "Delta Labs"], [(1, 2, "acquired")])
    s = score(pred, GOLD, embedder)
    assert s["entity_precision"] == pytest.approx(2 / 3)
    assert s["entity_recall"] == 1.0


def test_paraphrased_relation_matches(embedder):
    pred = extraction(["Acme Corp", "Gamma AI"], [(1, 2, "purchased")])
    s = score(pred, GOLD, embedder)
    assert s["edge_precision"] == 1.0
    assert s["edge_recall"] == 1.0


def test_dissimilar_relation_does_not_match(embedder):
    pred = extraction(["Acme Corp", "Gamma AI"], [(1, 2, "is headquartered in")])
    s = score(pred, GOLD, embedder)
    assert s["edge_precision"] == 0.0
    assert s["edge_recall"] == 0.0


def test_self_loop_edge_in_prediction_is_unmatched(embedder):
    pred = extraction(["Acme Corp", "Gamma AI"], [(1, 2, "acquired"), (2, 2, "acquired")])
    s = score(pred, GOLD, embedder)
    assert s["edge_precision"] == 0.5
    assert s["edge_recall"] == 1.0
