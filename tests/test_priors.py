"""Predicate-prior behavior, driven by a deterministic stub embedder."""

from collections.abc import Callable

import numpy as np
import pytest

from worldgraph.embed import Embedder
from worldgraph.graph import Graph, Statement
from worldgraph.priors import make_predicate_prior

_frame = "A {} B".format


class StubEmbedder(Embedder):
    """Returns an explicit predicate → vector dict, recording each call's
    (frame-transformed) keys."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, keys, template=None):
        self.calls.append([template(k) for k in keys] if template else list(keys))
        return {k: self._vectors[k] for k in keys}


def _graph(*predicates: str) -> Graph:
    graph = Graph()
    subject = graph.add_entity("subject")
    object_ = graph.add_entity("object")
    for predicate in predicates:
        graph.add_statement(subject, predicate, object_)
    return graph


def _statements(graph: Graph) -> dict[str, Statement]:
    return {
        term.predicates[0]: term
        for term in graph.terms.values()
        if isinstance(term, Statement)
    }


def _make(
    vectors: dict[str, np.ndarray], graphs: list[Graph]
) -> tuple[Callable[[Statement, Statement], float], StubEmbedder]:
    embedder = StubEmbedder(vectors)
    return make_predicate_prior(graphs, embedder), embedder


def test_identical_predicates_short_circuit_to_one():
    prior, embedder = _make(
        {"acquired": np.array([1.0, 0.0])},
        [_graph("acquired"), _graph("acquired")],
    )
    same_predicate = [
        t
        for t in _graph("acquired", "acquired").terms.values()
        if isinstance(t, Statement)
    ]
    first, second = same_predicate

    calls_before = len(embedder.calls)
    assert prior(first, second) == 1.0
    # Identical strings never touch the embedding machinery, not even the cache.
    assert len(embedder.calls) == calls_before

    def _unknown(id_: str) -> Statement:
        return Statement(id=id_, graph_id="g", subject="s", predicates=["merged"], object="o")

    assert prior(_unknown("u1"), _unknown("u2")) == 1.0
    assert len(embedder.calls) == calls_before


def test_monotone_above_baseline():
    prior, _ = _make(
        {
            "acquired": np.array([1.0, 0.0]),
            "bought out": np.array([0.72, np.sqrt(1.0 - 0.72**2)]),
            "invested in": np.array([0.86, np.sqrt(1.0 - 0.86**2)]),
            "partnered with": np.array([0.93, np.sqrt(1.0 - 0.93**2)]),
        },
        [_graph("acquired", "bought out", "invested in", "partnered with")],
    )
    s = _statements(_graph("acquired", "bought out", "invested in", "partnered with"))

    scores = [prior(s["acquired"], s[p]) for p in ("bought out", "invested in", "partnered with")]
    assert scores == sorted(scores)
    # Affine map from [0.65, 1.0] onto [0.5, 1.0]: cosine 0.72 -> 0.6, 0.86 -> 0.8, 0.93 -> 0.9.
    assert scores[0] == pytest.approx(0.6)
    assert scores[1] == pytest.approx(0.8)
    assert scores[2] == pytest.approx(0.9)


def test_at_or_below_baseline_is_neutral():
    prior, _ = _make(
        {
            "acquired": np.array([1.0, 0.0]),
            "located in": np.array([0.0, 1.0]),  # orthogonal, cosine 0
            "filed for": np.array([0.65, np.sqrt(1.0 - 0.65**2)]),  # cosine exactly at baseline
        },
        [_graph("acquired", "located in", "filed for")],
    )
    s = _statements(_graph("acquired", "located in", "filed for"))

    assert prior(s["acquired"], s["located in"]) == 0.5
    assert prior(s["acquired"], s["filed for"]) == 0.5


def test_unknown_predicate_at_call_time_is_neutral():
    prior, _ = _make(
        {"acquired": np.array([1.0, 0.0]), "purchased": np.array([0.9, 0.1])},
        [_graph("acquired", "purchased")],
    )
    s = _statements(_graph("acquired", "purchased"))
    unknown = Statement(id="u", graph_id="g", subject="s", predicates=["rumored about"], object="o")

    assert prior(unknown, s["acquired"]) == 0.5
    assert prior(s["purchased"], unknown) == 0.5


def test_prior_capped_at_one():
    # Non-unit parallel vectors push the raw dot product past 1.0.
    prior, _ = _make(
        {"acquired": np.array([2.0, 0.0]), "purchased": np.array([5.0, 0.0])},
        [_graph("acquired", "purchased")],
    )
    s = _statements(_graph("acquired", "purchased"))

    assert prior(s["acquired"], s["purchased"]) == 1.0


def test_each_distinct_predicate_embedded_exactly_once():
    prior, embedder = _make(
        {
            "acquired": np.array([1.0, 0.0]),
            "purchased": np.array([1.0, 0.0]),
            "located in": np.array([0.0, 1.0]),
        },
        [_graph("acquired", "located in"), _graph("acquired", "purchased", "acquired")],
    )
    assert embedder.calls == [
        [_frame(p) for p in ("acquired", "located in", "purchased")]
    ]

    s = _statements(_graph("acquired", "purchased", "located in"))
    prior(s["acquired"], s["purchased"])
    prior(s["acquired"], s["located in"])
    assert embedder.calls == [
        [_frame(p) for p in ("acquired", "located in", "purchased")]
    ]


def test_missing_embedding_model_raises(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(KeyError):
        make_predicate_prior([Graph()])
