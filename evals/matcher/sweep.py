"""Parameter sweep: run every scenario at every grid point.

A pair merges iff confidence ≥ merge_threshold AND at least one
structurally tested neighbor — names and priors alone never merge. The
sweep records, per scenario × grid point × focal pair, the confidence,
the merge verdict, and the margin to the bar, so the report can show
which parameter regions satisfy which intents. Fully deterministic:
scenarios, focal pairs, and grid points iterate in fixed order.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evals.matcher.cases import Intent, Scenario
from worldgraph.match import CALIBRATION, MatcherParams, match_graphs

Qid = tuple[str, str]


@dataclass(frozen=True)
class Grid:
    """Sweep space: the three data-decision axes, with the mechanical
    axes (damping, iteration budget, convergence) fixed at CALIBRATION's
    values."""

    merge_thresholds: tuple[float, ...]
    prior_strengths: tuple[float, ...]
    exp_lambdas: tuple[float, ...]
    damping: float = CALIBRATION.damping
    max_iter: int = CALIBRATION.max_iter
    epsilon: float = CALIBRATION.epsilon

    def points(self) -> tuple[MatcherParams, ...]:
        return tuple(
            MatcherParams(mt, ps, lam, self.damping, self.max_iter, self.epsilon)
            for mt in self.merge_thresholds
            for ps in self.prior_strengths
            for lam in self.exp_lambdas
        )


DEFAULT_GRID = Grid(
    merge_thresholds=(0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
    prior_strengths=(0.5, 1.0, 2.0, 3.0, 5.0),
    exp_lambdas=(1.0, 2.0, 4.0),
)


@dataclass(frozen=True)
class PairResult:
    """One focal pair's outcome at one parameter vector."""

    scenario_id: str
    label: str
    left: Qid
    right: Qid
    intent: Intent
    params: MatcherParams
    confidence: float
    merged: bool

    @property
    def margin(self) -> float:
        """Confidence minus the bar; how much room (or deficit) the pair
        had at this parameter vector."""
        return self.confidence - self.params.merge_threshold

    @property
    def held_at_gate(self) -> bool:
        """Confidence cleared the bar but no tested neighbor backed the
        pair — the structural-evidence gate held it apart."""
        return self.confidence >= self.params.merge_threshold and not self.merged

    @property
    def satisfied(self) -> bool | None:
        """None for boundary pairs (observed, never judged)."""
        if self.intent == "boundary":
            return None
        return self.merged == (self.intent == "merge")


def evaluate_scenario(scenario: Scenario, params: MatcherParams) -> list[PairResult]:
    """Run the matcher on one scenario under one parameter vector and
    record every focal pair's outcome."""
    confidence, groups, _merged = match_graphs(
        list(scenario.graphs), params=params, predicate_prior=scenario.predicate_prior
    )
    results: list[PairResult] = []
    for focal in scenario.focal_pairs:
        key = (focal.left, focal.right)
        if key not in confidence:
            raise ValueError(
                f"{scenario.id}: focal pair {key} has no confidence entry — "
                "harness bug: focal pairs must be cross-graph, same-kind, "
                "and reference existing term ids"
            )
        merged = any({focal.left, focal.right} <= set(group) for group in groups)
        results.append(
            PairResult(
                scenario_id=scenario.id,
                label=focal.label,
                left=focal.left,
                right=focal.right,
                intent=focal.intent,
                params=params,
                confidence=confidence[key],
                merged=merged,
            )
        )
    return results


def run_grid(scenarios: Sequence[Scenario], grid: Grid) -> list[PairResult]:
    """Every scenario at every grid point, in fixed order."""
    return [
        result
        for scenario in scenarios
        for params in grid.points()
        for result in evaluate_scenario(scenario, params)
    ]


def run_calibration(scenarios: Sequence[Scenario]) -> list[PairResult]:
    """Every scenario under CALIBRATION — the reference point the report
    marks, independent of whether CALIBRATION lies inside the grid."""
    return [
        result
        for scenario in scenarios
        for result in evaluate_scenario(scenario, CALIBRATION)
    ]
