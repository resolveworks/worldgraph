"""Extraction eval harness: pydantic-evals dataset runner over article fixtures.

Cases in ``datasets/extraction.yaml`` name fixture stems under ``fixtures/``
with a hand-authored expected extraction: entities by name, facts as
recursive triples — subject name or nested fact, predicate, object name or
nested fact. The task runs the production extraction path; the evaluator
scores entity and fact precision/recall against the golden (exact surface
match; term ids are never compared) with one assertion per expected
entity and fact, and the harness prints each case's misses (expected, not
extracted) and extras (extracted, not expected). The matcher is
deterministic and covered by unit tests — this suite judges only the
extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from worldgraph.extract import build_agent, extract_article, extraction_to_graph
from worldgraph.graph import Entity, Graph, Statement, Term

FIXTURES = Path(__file__).parent / "fixtures"
DATASET_PATH = Path(__file__).parent / "datasets" / "extraction.yaml"

MODEL = os.environ["EXTRACTION_MODEL"]


# ---------------------------------------------------------------------------
# Golden schema
# ---------------------------------------------------------------------------


class FactRef(BaseModel, extra="forbid"):
    """A fact located by surface form: ``predicate`` plus each endpoint's
    surface form — an entity name, or the referenced fact's own
    ``FactRef`` when the endpoint nests."""

    subject: str | FactRef
    predicate: str
    object: str | FactRef


Ref = str | FactRef


class ExpectedExtraction(BaseModel, extra="forbid"):
    """The golden: every entity the article names and every fact it
    asserts, in the expected extraction convention (short active-voice
    base-form predicates; qualifiers as statements about the statement
    they qualify)."""

    entities: list[str]
    facts: list[FactRef]


class ExtractionCase(BaseModel, extra="forbid"):
    """One article under test: the fixture stem and its golden."""

    article: str
    expected: ExpectedExtraction


# ---------------------------------------------------------------------------
# Task: the production extraction path on one article
# ---------------------------------------------------------------------------


def task(case: ExtractionCase) -> Graph:
    return extraction_to_graph(
        case.article,
        extract_article(
            build_agent(MODEL), (FIXTURES / f"{case.article}.md").read_text()
        ),
    )


# ---------------------------------------------------------------------------
# Surface-form comparison
# ---------------------------------------------------------------------------


def _matches(graph: Graph, term: Term, ref: Ref) -> bool:
    if isinstance(ref, str):
        return isinstance(term, Entity) and ref in term.names
    return (
        isinstance(term, Statement)
        and ref.predicate in term.predicates
        and _matches(graph, graph.terms[term.subject], ref.subject)
        and _matches(graph, graph.terms[term.object], ref.object)
    )


def render(ref: Ref) -> str:
    """Compact surface form: names plain, facts as
    ``subject —predicate→ object`` with nested facts bracketed."""
    if isinstance(ref, str):
        return ref

    def endpoint(e: Ref) -> str:
        return e if isinstance(e, str) else f"[{render(e)}]"

    return f"{endpoint(ref.subject)} —{ref.predicate}→ {endpoint(ref.object)}"


def render_term(graph: Graph, term: Term, stack: frozenset[str] = frozenset()) -> str:
    """Render an extracted term the same way; ``stack`` breaks reference
    cycles, which valid graphs may contain."""
    if isinstance(term, Entity):
        return " | ".join(term.names)
    if term.id in stack:
        return "⟲"

    inner = frozenset({*stack, term.id})

    def endpoint(term_id: str) -> str:
        rendered = render_term(graph, graph.terms[term_id], inner)
        return (
            f"[{rendered}]"
            if isinstance(graph.terms[term_id], Statement)
            else rendered
        )

    return f"{endpoint(term.subject)} —{term.predicates[0]}→ {endpoint(term.object)}"


@dataclass
class Comparison:
    """Precision/recall over entities and facts, plus both diff
    directions: misses (expected, not extracted) and extras (extracted,
    not expected)."""

    missed_entities: list[str]
    extra_entities: list[str]
    missed_facts: list[FactRef]
    extra_facts: list[str]  # rendered surface forms
    entity_precision: float
    entity_recall: float
    fact_precision: float
    fact_recall: float

    def clean(self) -> bool:
        return not (
            self.missed_entities
            or self.extra_entities
            or self.missed_facts
            or self.extra_facts
        )


def compare(graph: Graph, expected: ExpectedExtraction) -> Comparison:
    entities = [t for t in graph.terms.values() if isinstance(t, Entity)]
    statements = [t for t in graph.terms.values() if isinstance(t, Statement)]

    missed_entities = [
        name
        for name in expected.entities
        if not any(name in entity.names for entity in entities)
    ]
    extra_entities = [
        render_term(graph, entity)
        for entity in entities
        if not any(name in expected.entities for name in entity.names)
    ]
    missed_facts = [
        fact
        for fact in expected.facts
        if not any(_matches(graph, statement, fact) for statement in statements)
    ]
    extra_facts = [
        render_term(graph, statement)
        for statement in statements
        if not any(_matches(graph, statement, fact) for fact in expected.facts)
    ]

    def ratio(part: int, whole: int) -> float:
        return part / whole if whole else 1.0

    return Comparison(
        missed_entities=missed_entities,
        extra_entities=extra_entities,
        missed_facts=missed_facts,
        extra_facts=extra_facts,
        entity_precision=ratio(len(entities) - len(extra_entities), len(entities)),
        entity_recall=ratio(len(expected.entities) - len(missed_entities), len(expected.entities)),
        fact_precision=ratio(len(statements) - len(extra_facts), len(statements)),
        fact_recall=ratio(len(expected.facts) - len(missed_facts), len(expected.facts)),
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class ExtractionHit(Evaluator[ExtractionCase, Graph, object]):
    """One assertion per expected entity and fact (recall granularity),
    plus entity and fact precision/recall as scores."""

    def evaluate(
        self, ctx: EvaluatorContext[ExtractionCase, Graph, object]
    ) -> dict[str, bool | float]:
        expected = ctx.inputs.expected
        result = compare(ctx.output, expected)
        return {
            **{
                f"entity: {name}": name not in result.missed_entities
                for name in expected.entities
            },
            **{
                f"fact: {render(fact)}": fact not in result.missed_facts
                for fact in expected.facts
            },
            "entity_precision": round(result.entity_precision, 3),
            "entity_recall": round(result.entity_recall, 3),
            "fact_precision": round(result.fact_precision, 3),
            "fact_recall": round(result.fact_recall, 3),
        }


def main() -> None:
    dataset = Dataset[ExtractionCase, Graph, object].from_file(DATASET_PATH)
    dataset.evaluators.append(ExtractionHit())
    report = dataset.evaluate_sync(task)
    report.print()

    for case in report.cases:
        result = compare(case.output, case.inputs.expected)
        if result.clean():
            continue
        print(f"\n{case.name}:")
        print("  entities -:", *(result.missed_entities or ["·"]), sep="\n    ")
        print("  entities +:", *(result.extra_entities or ["·"]), sep="\n    ")
        print(
            "  facts    -:",
            *([render(fact) for fact in result.missed_facts] or ["·"]),
            sep="\n    ",
        )
        print("  facts    +:", *(result.extra_facts or ["·"]), sep="\n    ")


if __name__ == "__main__":
    main()
