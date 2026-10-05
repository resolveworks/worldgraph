"""Markdown report rendering for the matcher parameter sweep.

Three views: (1) per-scenario outcomes under CALIBRATION and across the
grid; (2) the boundary view — which grid points satisfy every claimed
intent simultaneously, or the finding that none do; (3) boundary-tagged
focal pairs' merge distribution across the grid. Rendering is a pure
function of (scenarios, grid, results) — no timestamps, sorted
iteration, identical runs produce identical reports.
"""

from collections.abc import Sequence

from evals.matcher.cases import CLAIMED_INTENTS, Scenario
from evals.matcher.sweep import Grid, PairResult
from worldgraph.match import CALIBRATION, MatcherParams


def _f(value: float) -> str:
    return f"{value:.3f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%"


def _calibration_cell(result: PairResult) -> str:
    star = "★"
    if result.merged:
        return f"merge {star} (conf {_f(result.confidence)}, margin {_signed(result.margin)})"
    if result.held_at_gate:
        return f"apart {star} — held at gate: conf {_f(result.confidence)} ≥ bar, no tested neighbor"
    return (
        f"apart {star} (conf {_f(result.confidence)}, margin {_signed(result.margin)})"
    )


def _header(grid: Grid, n_points: int, n_claimed: int) -> list[str]:
    grid_shape = (
        f"{len(grid.merge_thresholds)} × {len(grid.prior_strengths)} "
        f"× {len(grid.exp_lambdas)} = {n_points} points"
    )
    return [
        "# Matcher eval — parameter sweep report",
        "",
        "Tests prove the matcher's mechanism and quantify over parameters;",
        "this report makes the parameter VALUES visible: which scenarios merge",
        "under which regions of the sweep, and with how much margin. A pair",
        "merges iff confidence ≥ merge_threshold AND at least one structurally",
        "tested neighbor — names and priors alone never merge. Intents",
        "`merge`/`apart` are claims the calibration should satisfy; `boundary`",
        "pairs are genuinely unknowable at their corpus size — observed, never",
        "judged. Unsatisfied intents are findings, not harness failures.",
        "",
        f"Grid: {grid_shape} — merge_threshold {list(grid.merge_thresholds)},",
        f"prior_strength {list(grid.prior_strengths)},",
        f"exp_lambda {list(grid.exp_lambdas)};",
        f"damping {grid.damping}, max_iter {grid.max_iter},",
        f"epsilon {grid.epsilon} (CALIBRATION values).",
        (
            f"★ marks CALIBRATION "
            f"(merge_threshold={CALIBRATION.merge_threshold}, "
            f"prior_strength={CALIBRATION.prior_strength}, "
            f"exp_lambda={CALIBRATION.exp_lambda})."
        ),
        f"{n_claimed} claimed focal pairs across the scenarios.",
        "",
    ]


def _scenario_section(
    scenario: Scenario,
    results: list[PairResult],
    calibration: dict[tuple[str, tuple[str, str], tuple[str, str]], PairResult],
    n_points: int,
) -> list[str]:
    lines = [f"### {scenario.id}", "", scenario.description, ""]
    lines.append(
        "| focal pair | intent | CALIBRATION | intent satisfied | merged |"
        " confidence across grid |"
    )
    lines.append("|---|---|---|---|---|---|")
    for focal in scenario.focal_pairs:
        pair_results = [
            r for r in results if r.left == focal.left and r.right == focal.right
        ]
        cal = calibration[(scenario.id, focal.left, focal.right)]
        confs = [r.confidence for r in pair_results]
        merged_fraction = _pct(sum(r.merged for r in pair_results), n_points)
        if focal.intent == "boundary":
            satisfied = "—"
        else:
            satisfied = _pct(sum(bool(r.satisfied) for r in pair_results), n_points)
        lines.append(
            f"| {focal.label} | {focal.intent} | {_calibration_cell(cal)} "
            f"| {satisfied} | {merged_fraction} "
            f"| [{_f(min(confs))}, {_f(max(confs))}] |"
        )
    lines.append("")
    return lines


def _region_section(
    grid: Grid,
    results: Sequence[PairResult],
    n_points: int,
) -> list[str]:
    claimed = [r for r in results if r.intent in CLAIMED_INTENTS]
    by_point: dict[MatcherParams, list[PairResult]] = {}
    for result in claimed:
        by_point.setdefault(result.params, []).append(result)
    passing = {
        params: all(bool(r.satisfied) for r in point_results)
        for params, point_results in by_point.items()
    }
    n_pass = sum(passing.values())

    lines = ["## Grid regions satisfying every claimed intent", ""]
    if n_pass == 0:
        lines.append(
            "**EMPTY**: no parameter vector in the grid satisfies all claimed"
            " intents simultaneously — that itself is the finding. See the"
            " per-intent coverage below and the blockers list."
        )
        lines.append("")
    else:
        pass_points = [p for p, ok in passing.items() if ok]
        mts = sorted({p.merge_threshold for p in pass_points})
        pss = sorted({p.prior_strength for p in pass_points})
        lams = sorted({p.exp_lambda for p in pass_points})
        lines.append(
            f"{n_pass} of {n_points} grid points satisfy every claimed"
            " intent. Satisfying region spans:"
            f" merge_threshold ∈ {mts}, prior_strength ∈ {pss},"
            f" exp_lambda ∈ {lams}."
        )
        lines.append("")

    for lam in grid.exp_lambdas:
        lines.append(f"#### exp_lambda = {lam}")
        lines.append("")
        header = (
            "| merge_threshold \\ prior_strength | "
            + " | ".join(_f(ps) for ps in grid.prior_strengths)
            + " |"
        )
        lines.append(header)
        lines.append("|---" * (len(grid.prior_strengths) + 1) + "|")
        for mt in grid.merge_thresholds:
            cells = []
            for ps in grid.prior_strengths:
                params = MatcherParams(
                    mt, ps, lam, grid.damping, grid.max_iter, grid.epsilon
                )
                mark = "✓" if passing.get(params) else "·"
                if params == CALIBRATION:
                    mark += "★"
                cells.append(mark)
            lines.append(f"| {_f(mt)} | " + " | ".join(cells) + " |")
        lines.append("")

    # Blockers: claimed intents that fail at CALIBRATION, or everywhere.
    cal_failures = [
        r for r in claimed if r.params == CALIBRATION and r.satisfied is False
    ]
    lines.append("Unsatisfied at CALIBRATION:")
    if cal_failures:
        for r in cal_failures:
            verdict = "merged" if r.merged else "stayed apart"
            lines.append(
                f"- {r.scenario_id} / {r.label}: intent {r.intent},"
                f" {verdict} at conf {_f(r.confidence)}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("Never satisfied anywhere in the grid:")
    never = []
    for params_results in _group_by_pair(claimed).values():
        if not any(bool(r.satisfied) for r in params_results):
            never.append(params_results[0])
    if never:
        for r in never:
            lines.append(f"- {r.scenario_id} / {r.label}: intent {r.intent}")
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _group_by_pair(
    results: Sequence[PairResult],
) -> dict[tuple[str, str, tuple[str, str], tuple[str, str]], list[PairResult]]:
    grouped: dict[
        tuple[str, str, tuple[str, str], tuple[str, str]], list[PairResult]
    ] = {}
    for result in results:
        key = (result.scenario_id, result.label, result.left, result.right)
        grouped.setdefault(key, []).append(result)
    return grouped


def _boundary_section(
    scenarios: Sequence[Scenario],
    results: Sequence[PairResult],
    grid: Grid,
) -> list[str]:
    lines = [
        "## Boundary pairs across the grid",
        "",
        (
            "Merged percentage at each merge_threshold (over all prior_strength ×"
            " exp_lambda at that threshold) — where the empirically open "
            "boundary falls as the bar moves."
        ),
        "",
    ]
    header = (
        "| scenario | focal pair | "
        + " | ".join(_f(mt) for mt in grid.merge_thresholds)
        + " | overall merged | confidence |"
    )
    lines.append(header)
    lines.append("|---" * (len(grid.merge_thresholds) + 3) + "|")
    for scenario in scenarios:
        for focal in scenario.focal_pairs:
            if focal.intent != "boundary":
                continue
            pair_results = [
                r
                for r in results
                if r.scenario_id == scenario.id
                and r.left == focal.left
                and r.right == focal.right
            ]
            cells = []
            for mt in grid.merge_thresholds:
                at_mt = [r for r in pair_results if r.params.merge_threshold == mt]
                cells.append(_pct(sum(r.merged for r in at_mt), len(at_mt)))
            overall = _pct(sum(r.merged for r in pair_results), len(pair_results))
            confs = [r.confidence for r in pair_results]
            lines.append(
                f"| {scenario.id} | {focal.label} | "
                + " | ".join(cells)
                + f" | {overall} | [{_f(min(confs))}, {_f(max(confs))}] |"
            )
    lines.append("")
    return lines


def render_report(
    scenarios: Sequence[Scenario],
    grid: Grid,
    results: Sequence[PairResult],
    calibration: Sequence[PairResult],
) -> str:
    """Render the full markdown report. Pure and deterministic."""
    n_points = len(grid.points())
    calibration_by_pair = {(r.scenario_id, r.left, r.right): r for r in calibration}
    n_claimed = sum(
        1 for s in scenarios for f in s.focal_pairs if f.intent in CLAIMED_INTENTS
    )

    lines = _header(grid, n_points, n_claimed)
    lines.append("## Per-scenario outcomes")
    lines.append("")
    for scenario in scenarios:
        scenario_results = [r for r in results if r.scenario_id == scenario.id]
        lines.extend(
            _scenario_section(scenario, scenario_results, calibration_by_pair, n_points)
        )
    lines.extend(_region_section(grid, results, n_points))
    lines.extend(_boundary_section(scenarios, results, grid))
    lines.append("---")
    lines.append(
        "Regenerate: `uv run python -m evals.matcher.run` (deterministic;"
        " writes evals/matcher/report.md, which is gitignored)."
    )
    return "\n".join(lines) + "\n"
