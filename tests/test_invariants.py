"""Axiomatic properties of the matching pipeline.

Scenario tests (test_propagation, test_integration) pin specific matching
behaviors.  These tests pin the invariants every scenario implicitly
relies on, plus behavior on degenerate inputs:

- Symmetry: confidence is identical for both pair orderings
- Permutation invariance: graph input order cannot change results
- Determinism: identical inputs produce identical outputs
- Degenerate inputs: empty corpora, edgeless graphs, self-loops, and
  duplicate edges neither crash nor produce spurious merges
"""

import itertools

import pytest

from worldgraph.graph import Graph
from worldgraph.match import build_match_groups, match_graphs
from worldgraph.names import build_idf, soft_tfidf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_scenario() -> list[Graph]:
    """Three graphs with two real merges (identical names + shared
    structure) and one non-merge cluster.

    Exercises the full pipeline: seeding, propagation, progressive
    merging, and final grouping — so the invariance tests below are not
    vacuously satisfied by a trivial run.
    """
    graphs = []
    for i, (acquirer, target, rel) in enumerate(
        [
            ("Meridian Corp", "DataVault", "acquired"),
            ("Meridian Corp", "DataVault", "purchased"),
            ("NexGen Holdings", "ClearSky", "acquired"),
        ]
    ):
        g = Graph(id=f"g{i}")
        acq = g.add_entity(acquirer)
        tgt = g.add_entity(target)
        ceo = g.add_entity("James Chen" if i < 2 else "Sarah Park")
        g.add_edge(acq, tgt, rel)
        g.add_edge(acq, ceo, "CEO is")
        graphs.append(g)
    return graphs


def _group_sets(graphs: list[Graph], embedder) -> set[frozenset[str]]:
    confidence = match_graphs(graphs, embedder)
    groups, _ = build_match_groups(graphs, confidence)
    return {frozenset(g) for g in groups}


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_confidence_is_symmetric(embedder):
    """For every pair key, the reversed key must exist with the same score.

    Consumers index confidence with arbitrary pair orderings; an
    asymmetric dict would silently return 0.0 for half of all lookups."""
    graphs = _merge_scenario()
    confidence = match_graphs(graphs, embedder)

    assert confidence, "scenario produced no confidence entries"
    for (a, b), score in confidence.items():
        assert (b, a) in confidence, f"missing reverse key for {(a, b)}"
        assert confidence[(b, a)] == score, f"asymmetric score for {(a, b)}"


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------


def test_graph_order_does_not_change_results(embedder):
    """The input order of graphs must not affect confidence or groups.

    Pair ordering, adjacency construction, and float summation order all
    derive from the input list; none of them may leak into the result."""
    graphs = _merge_scenario()

    base_conf = match_graphs(graphs, embedder)
    base_groups = _group_sets(graphs, embedder)

    # Premise: the scenario actually merges — otherwise this test is vacuous
    assert len(base_groups) >= 2, f"scenario too weak, only got {base_groups}"

    for perm in itertools.permutations(graphs):
        perm_conf = match_graphs(list(perm), embedder)
        perm_groups = _group_sets(list(perm), embedder)

        assert perm_groups == base_groups, f"groups differ for order {[g.id for g in perm]}"
        assert set(perm_conf) == set(base_conf)
        for key, score in base_conf.items():
            assert abs(perm_conf[key] - score) < 1e-9, f"score drift for {key}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_matching_is_deterministic(embedder):
    """Running the same input twice must produce identical output.

    No RNG, no wall-clock, and no set-iteration-order dependence is
    allowed anywhere in the pipeline."""
    graphs = _merge_scenario()
    assert match_graphs(graphs, embedder) == match_graphs(graphs, embedder)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_input_returns_no_matches(embedder):
    """An empty graph list is valid input and yields no matches."""
    assert match_graphs([], embedder) == {}

    groups, unified = build_match_groups([], {})
    assert groups == []
    assert len(unified.nodes) == 0


def test_edgeless_identical_names_score_at_seed(embedder):
    """Edgeless entities fall back to the name-similarity seed.

    Per docs/negative_evidence.md: "With no structural evidence the score
    equals the seed."  Identical names therefore score 1.0 — but merging
    is a separate question (see test_name_similarity_alone_never_merges)."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")

    confidence = match_graphs([g1, g2], embedder)

    assert confidence[(apple1.id, apple2.id)] == pytest.approx(1.0)
    assert confidence[(beats1.id, beats2.id)] == pytest.approx(1.0)


@pytest.mark.xfail(
    reason="build_match_groups unions on score >= threshold without the "
    "n_tested > 0 structural-evidence gate from propagate_similarity, so "
    "identical names merge despite docs/negative_evidence.md: 'name "
    "similarity alone never triggers a merge'",
    strict=True,
)
def test_name_similarity_alone_never_merges(embedder):
    """Dangling entities with identical names must not form match groups.

    docs/negative_evidence.md: the score falls back to the name-similarity
    seed, "but the n_tested > 0 merge gate prevents merging" — name
    similarity alone must never produce a group."""
    g1 = Graph(id="g1")
    g1.add_entity("Apple")
    g1.add_entity("Beats")

    g2 = Graph(id="g2")
    g2.add_entity("Apple")
    g2.add_entity("Beats")

    confidence = match_graphs([g1, g2], embedder)
    groups, _ = build_match_groups([g1, g2], confidence)

    assert groups == [], f"name-only merge: {groups}"


def test_self_loops_and_duplicate_edges_are_harmless(embedder):
    """Self-loops and duplicate edges must neither crash matching nor
    distort it.

    The duplicate edge counts once (adjacency deduplication) and the
    self-loop contributes no evidence — matching behaves as if only the
    clean X→Y edge existed."""
    g1 = Graph(id="g1")
    x1 = g1.add_entity("X")
    y1 = g1.add_entity("Y")
    g1.add_edge(x1, x1, "acquired")  # self-loop
    g1.add_edge(x1, y1, "acquired")
    g1.add_edge(x1, y1, "acquired")  # duplicate

    g2 = Graph(id="g2")
    x2 = g2.add_entity("X")
    y2 = g2.add_entity("Y")
    g2.add_edge(x2, y2, "acquired")

    confidence = match_graphs([g1, g2], embedder)
    groups, _ = build_match_groups([g1, g2], confidence)

    assert {frozenset(g) for g in groups} == {
        frozenset({x1.id, x2.id}),
        frozenset({y1.id, y2.id}),
    }
