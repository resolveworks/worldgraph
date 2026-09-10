"""Axiomatic properties of the matching pipeline.

Scenario tests (test_propagation, test_integration, test_event_matching)
pin specific matching behaviors.  These tests pin the invariants every
scenario implicitly relies on, plus behavior on degenerate inputs:

- Symmetry: confidence is identical for both pair orderings
- Permutation invariance: graph input order cannot change results
- Determinism: identical inputs produce identical outputs
- Degenerate inputs: empty corpora, edgeless graphs, unconnected events,
  and duplicate edges neither crash nor produce spurious merges
"""

import itertools

import pytest
from conftest import fact

from worldgraph.graph import Graph
from worldgraph.match import match_graphs

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
    for i, (acquirer, target, label) in enumerate(
        [
            ("Meridian Corp", "DataVault", "acquire"),
            ("Meridian Corp", "DataVault", "purchase"),
            ("NexGen Holdings", "ClearSky", "acquire"),
        ]
    ):
        g = Graph(id=f"g{i}")
        acq = g.add_entity(acquirer)
        tgt = g.add_entity(target)
        ceo = g.add_entity("James Chen" if i < 2 else "Sarah Park")
        fact(g, label, agent=acq, patients=(tgt,))
        fact(g, "employ as CEO", agent=acq, patients=(ceo,))
        graphs.append(g)
    return graphs


def _group_sets(graphs: list[Graph]) -> set[frozenset[str]]:
    _, groups, _ = match_graphs(graphs)
    return {frozenset(g) for g in groups}


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_confidence_is_symmetric():
    """For every pair key, the reversed key must exist with the same score.

    Consumers index confidence with arbitrary pair orderings; an
    asymmetric dict would silently return 0.0 for half of all lookups."""
    graphs = _merge_scenario()
    confidence, _, _ = match_graphs(graphs)

    assert confidence, "scenario produced no confidence entries"
    for (a, b), score in confidence.items():
        assert (b, a) in confidence, f"missing reverse key for {(a, b)}"
        assert confidence[(b, a)] == score, f"asymmetric score for {(a, b)}"


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------


def test_graph_order_does_not_change_results():
    """The input order of graphs must not affect confidence or groups.

    Pair ordering, adjacency construction, and float summation order all
    derive from the input list; none of them may leak into the result."""
    graphs = _merge_scenario()

    base_conf, _, _ = match_graphs(graphs)
    base_groups = _group_sets(graphs)

    # Premise: the scenario actually merges — otherwise this test is vacuous
    assert len(base_groups) >= 2, f"scenario too weak, only got {base_groups}"

    for perm in itertools.permutations(graphs):
        perm_conf, _, _ = match_graphs(list(perm))
        perm_groups = _group_sets(list(perm))

        assert perm_groups == base_groups, f"groups differ for order {[g.id for g in perm]}"
        assert set(perm_conf) == set(base_conf)
        for key, score in base_conf.items():
            assert abs(perm_conf[key] - score) < 1e-9, f"score drift for {key}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_matching_is_deterministic():
    """Running the same input twice must produce identical output.

    No RNG, no wall-clock, and no set-iteration-order dependence is
    allowed anywhere in the pipeline — including the unified graph id
    and the match groups, not just the confidence scores."""
    graphs = _merge_scenario()
    assert match_graphs(graphs) == match_graphs(graphs)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_input_returns_no_matches():
    """An empty graph list is valid input and yields no matches."""
    confidence, groups, unified = match_graphs([])
    assert confidence == {}
    assert groups == []
    assert len(unified.nodes) == 0


def test_edgeless_identical_names_score_at_seed():
    """Edgeless entities fall back to the name-similarity seed.

    With no structural evidence the score equals the seed.  Identical
    names therefore score 1.0 — but merging is a separate question
    (see test_name_similarity_alone_never_merges)."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")

    confidence, _, _ = match_graphs([g1, g2])

    assert confidence[(apple1.id, apple2.id)] == pytest.approx(1.0)
    assert confidence[(beats1.id, beats2.id)] == pytest.approx(1.0)


def test_name_similarity_alone_never_merges():
    """Dangling entities with identical names must not form match groups.

    With no structural evidence the score falls back to the name-similarity
    seed, but the n_tested > 0 merge gate prevents merging — name
    similarity alone must never produce a group. The in-loop union-find
    is the sole merge authority; there is no post-hoc grouping path that
    could bypass the structural-evidence gate."""
    g1 = Graph(id="g1")
    g1.add_entity("Apple")
    g1.add_entity("Beats")

    g2 = Graph(id="g2")
    g2.add_entity("Apple")
    g2.add_entity("Beats")

    _, groups, _ = match_graphs([g1, g2])

    assert groups == [], f"name-only merge: {groups}"


def test_unconnected_events_never_merge():
    """Event nodes without participants have no structural evidence and
    stay at the neutral prior — identical labels are never compared."""
    g1 = Graph(id="g1")
    event1 = g1.add_event("acquire")

    g2 = Graph(id="g2")
    event2 = g2.add_event("acquire")

    confidence, groups, _ = match_graphs([g1, g2])

    assert confidence[(event1.id, event2.id)] == pytest.approx(0.5)
    assert groups == []


def test_duplicate_edges_are_harmless():
    """Duplicate participation edges must neither crash matching nor
    distort it: the duplicate counts once (adjacency deduplication), and
    matching behaves as if only the clean edge existed."""
    g1 = Graph(id="g1")
    x1 = g1.add_entity("X")
    y1 = g1.add_entity("Y")
    event1 = g1.add_event("acquire")
    g1.add_edge(event1, x1, "agent")
    g1.add_edge(event1, y1, "patient")
    g1.add_edge(event1, y1, "patient")  # duplicate

    g2 = Graph(id="g2")
    x2 = g2.add_entity("X")
    y2 = g2.add_entity("Y")
    event2 = fact(g2, "acquire", agent=x2, patients=(y2,))

    _, groups, _ = match_graphs([g1, g2])

    assert {frozenset(g) for g in groups} == {
        frozenset({x1.id, x2.id}),
        frozenset({y1.id, y2.id}),
        frozenset({event1.id, event2.id}),
    }
