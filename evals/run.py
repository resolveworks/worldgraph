"""Extraction eval harness: pydantic-evals dataset runner over article fixtures.

Cases in ``datasets/extraction.yaml`` name fixture stems under ``fixtures/``.
The task runs the production extraction path on each article; the evaluator
asserts an exact hit on the hand-labeled golden extraction.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from worldgraph.extract import Extraction, build_agent, extract_article

load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"
DATASET_PATH = Path(__file__).parent / "datasets" / "extraction.yaml"

MODEL = "deepseek:deepseek-v4-flash"  # keep in sync with worldgraph/cli.py default


def task(stem: str) -> Extraction:
    """Run the production extraction path on a fixture article."""
    return extract_article(build_agent(MODEL), (FIXTURES / f"{stem}.md").read_text())


def canonical(ext: Extraction) -> tuple[Counter[str], Counter[tuple[str, str, str, str]]]:
    """Order- and id-independent form: entity name counts and
    (source, relation, target, temporal) name-tuple counts."""
    names = {e.id: e.name for e in ext.entities}
    if len(names) != len(ext.entities):
        raise ValueError(f"duplicate entity ids: {ext.entities!r}")
    return (
        Counter(e.name for e in ext.entities),
        Counter(
            (names[r.source], r.relation, names[r.target], r.temporal)
            for r in ext.relations
        ),
    )


@dataclass
class ExactHit(Evaluator[Extraction, Extraction, object]):
    """Pass iff the extraction is an exact hit on the golden — same entities,
    same facts, no extras, no paraphrases."""

    def evaluate(self, ctx: EvaluatorContext[Extraction, Extraction, object]) -> bool:
        assert ctx.expected_output is not None
        return canonical(ctx.output) == canonical(ctx.expected_output)


def main() -> None:
    dataset = Dataset[str, Extraction, object].from_file(DATASET_PATH)
    dataset.evaluators.append(ExactHit())
    report = dataset.evaluate_sync(task)
    report.print()

    for case in report.cases:
        if case.assertions["ExactHit"].value:
            continue
        gold_names, gold_triples = canonical(case.expected_output)
        pred_names, pred_triples = canonical(case.output)
        print(f"\n{case.name}: MISS")
        print("  entities -:", sorted((gold_names - pred_names).elements()) or "·",
              "+:", sorted((pred_names - gold_names).elements()) or "·")
        print("  triples  -:", sorted((gold_triples - pred_triples).elements()) or "·",
              "+:", sorted((pred_triples - gold_triples).elements()) or "·")


if __name__ == "__main__":
    main()
