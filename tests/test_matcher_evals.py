"""Smoke test for the matcher eval harness (evals/matcher/).

Guards infrastructure only: the scenario corpus is well-formed, the
sweep runs on a reduced grid, and the report renders deterministically
with every scenario present. It asserts nothing about matcher outcomes —
those belong to the eval report, not to tests (the two-layer doctrine:
tests prove the mechanism; evals make the parameter values visible).
"""

from evals.matcher.cases import CLAIMED_INTENTS, SCENARIOS
from evals.matcher.report import render_report
from evals.matcher.sweep import Grid, run_calibration, run_grid

# Reduced grid: 2 × 2 × 1 = 4 points — enough to exercise the harness,
# small enough to keep the smoke test fast.
REDUCED_GRID = Grid(
    merge_thresholds=(0.6, 0.8),
    prior_strengths=(1.0, 3.0),
    exp_lambdas=(2.0,),
)


def test_scenarios_are_well_formed():
    """Every focal pair references existing terms in distinct graphs, and
    every intent is one of the three tags — the harness's inputs."""
    ids = [scenario.id for scenario in SCENARIOS]
    assert len(ids) == len(set(ids))
    assert SCENARIOS
    for scenario in SCENARIOS:
        assert scenario.graphs, scenario.id
        qids = {
            (graph.id, term_id) for graph in scenario.graphs for term_id in graph.terms
        }
        graph_ids = {graph.id for graph in scenario.graphs}
        assert len(graph_ids) == len(scenario.graphs), scenario.id
        assert scenario.focal_pairs, scenario.id
        for focal in scenario.focal_pairs:
            assert focal.left in qids, (scenario.id, focal.label)
            assert focal.right in qids, (scenario.id, focal.label)
            assert focal.left[0] != focal.right[0], (scenario.id, focal.label)
            assert focal.intent in {*CLAIMED_INTENTS, "boundary"}, (
                scenario.id,
                focal.label,
            )


def test_sweep_runs_on_reduced_grid():
    """One result per scenario × focal pair × grid point, covering every
    scenario id."""
    results = run_grid(SCENARIOS, REDUCED_GRID)

    n_focal = sum(len(scenario.focal_pairs) for scenario in SCENARIOS)
    assert len(results) == n_focal * len(REDUCED_GRID.points())
    assert {r.scenario_id for r in results} == {s.id for s in SCENARIOS}
    assert all(0.0 <= r.confidence <= 1.0 for r in results)


def test_report_is_deterministic_and_complete():
    """Identical runs render identical reports containing every scenario,
    focal pair, and calibration reference."""
    calibration = run_calibration(SCENARIOS)
    first = render_report(
        SCENARIOS, REDUCED_GRID, run_grid(SCENARIOS, REDUCED_GRID), calibration
    )
    second = render_report(
        SCENARIOS, REDUCED_GRID, run_grid(SCENARIOS, REDUCED_GRID), calibration
    )

    assert first == second
    for scenario in SCENARIOS:
        assert f"### {scenario.id}" in first
        for focal in scenario.focal_pairs:
            assert focal.label in first
    assert "★" in first  # CALIBRATION reference point is marked
    assert "CALIBRATION" in first
