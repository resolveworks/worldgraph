"""Run the matcher parameter sweep and write the report.

Usage::

    uv run python -m evals.matcher.run

Fully deterministic; no LLM or embedding calls (statement pairs seed via
the neutral prior or per-scenario deterministic stub priors). Prints the
report to stdout and writes it to ``evals/matcher/report.md`` — the
report is a generated artifact, gitignored and regenerated on demand,
never committed stale.
"""

from pathlib import Path

from evals.matcher.cases import SCENARIOS
from evals.matcher.report import render_report
from evals.matcher.sweep import DEFAULT_GRID, run_calibration, run_grid

REPORT_PATH = Path(__file__).parent / "report.md"


def main() -> None:
    results = run_grid(SCENARIOS, DEFAULT_GRID)
    calibration = run_calibration(SCENARIOS)
    report = render_report(SCENARIOS, DEFAULT_GRID, results, calibration)
    REPORT_PATH.write_text(report)
    print(report, end="")
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
