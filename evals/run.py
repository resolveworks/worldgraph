"""Extraction eval harness: pydantic-evals dataset runner over article fixtures.

Cases in ``datasets/extraction.yaml`` name fixture stems under ``fixtures/``.
The task runs the production extraction path on each article; the evaluator
asserts an exact hit on the hand-labeled golden extraction.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from worldgraph.extract import Event, Extraction, build_agent, extract_article

load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"
DATASET_PATH = Path(__file__).parent / "datasets" / "extraction.yaml"

MODEL = os.environ["EXTRACTION_MODEL"]


def task(stem: str) -> Extraction:
    """Run the production extraction path on a fixture article."""
    return extract_article(build_agent(MODEL), (FIXTURES / f"{stem}.md").read_text())


def canonical(ext: Extraction) -> tuple[Counter[str], Counter[tuple]]:
    """Order- and id-independent form: entity name counts and event form
    counts. An event form is ``(label, ((role, term), ...))`` with
    participants sorted; a term is an entity name or, where a participant
    references an event, that event's own form — so a nested participation
    is compared against the specific event it references, never its label
    alone.
    """
    names = {entity.id: entity.name for entity in ext.entities}
    events = {event.id: event for event in ext.events}

    def term(ref: str, stack: tuple[str, ...]) -> str | tuple:
        if ref in names:
            return names[ref]
        if ref in stack:
            raise ValueError(f"cyclic event references: {ref!r} in {stack + (ref,)}")
        return form(events[ref], stack + (ref,))

    def form(event: Event, stack: tuple[str, ...]) -> tuple:
        participants = tuple(
            sorted(
                ((p.role, term(p.ref, stack)) for p in event.participants),
                key=repr,
            )
        )
        return (event.label, participants)

    return (
        Counter(names.values()),
        Counter(form(event, ()) for event in ext.events),
    )


def render(term: str | tuple) -> str:
    """Human-readable term: an entity name plain, an event form as
    ``label(role=term, ...)`` with nested forms bracketed."""
    if isinstance(term, str):
        return term
    label, participants = term

    def show(t: str | tuple) -> str:
        return t if isinstance(t, str) else f"[{render(t)}]"

    inner = ", ".join(f"{role}={show(t)}" for role, t in participants)
    return f"{label}({inner})"


@dataclass
class ExactHit(Evaluator[Extraction, Extraction, object]):
    """Pass iff the extraction is an exact hit on the golden — same entities,
    same events with their participant structure, no extras, no paraphrases."""

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
        gold_names, gold_forms = canonical(case.expected_output)
        pred_names, pred_forms = canonical(case.output)
        missing = sorted(render(f) for f in (gold_forms - pred_forms).elements())
        extra = sorted(render(f) for f in (pred_forms - gold_forms).elements())
        print(f"\n{case.name}: MISS")
        print(
            "  entities  -:",
            sorted((gold_names - pred_names).elements()) or "·",
            "+:",
            sorted((pred_names - gold_names).elements()) or "·",
        )
        print("  events    -:", *(missing or ["·"]), sep="\n    ")
        print("  events    +:", *(extra or ["·"]), sep="\n    ")


if __name__ == "__main__":
    main()
