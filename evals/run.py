"""Extraction eval harness over self-contained YAML cases.

Each file under ``cases/`` contains an article and its expected term graph;
the filename is the case name. Golden ids express term identity and shared
references but are local to the golden: comparison finds the largest
structurally consistent mapping to the extracted graph, so ids are never
compared. The evaluator scores entity and statement precision/recall and
prints misses and extras in both directions. The matcher is deterministic
and covered by unit tests — this suite judges only extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, model_validator
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from worldgraph.extract import build_agent, extract_article, extraction_to_graph
from worldgraph.graph import Entity, Graph, Statement, Term

CASES = Path(__file__).parent / "cases"

MODEL = os.environ["EXTRACTION_MODEL"]


# ---------------------------------------------------------------------------
# Golden schema
# ---------------------------------------------------------------------------


class ExpectedEntity(BaseModel, extra="forbid"):
    id: str
    name: str


class ExpectedStatement(BaseModel, extra="forbid"):
    id: str
    subject: str
    predicate: str
    object: str


ExpectedTerm = ExpectedEntity | ExpectedStatement


class ExpectedExtraction(BaseModel, extra="forbid"):
    """A golden term graph whose ids define identity only within the case."""

    entities: list[ExpectedEntity]
    statements: list[ExpectedStatement]

    @model_validator(mode="after")
    def _validate_graph(self) -> ExpectedExtraction:
        ids = [term.id for term in self.entities + self.statements]
        duplicates = sorted({term_id for term_id in ids if ids.count(term_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate golden term ids: {duplicates}")

        known = set(ids)
        unknown = sorted(
            {
                endpoint
                for statement in self.statements
                for endpoint in (statement.subject, statement.object)
                if endpoint not in known
            }
        )
        if unknown:
            raise ValueError(f"golden statements reference unknown ids: {unknown}")

        self_refs = sorted(
            statement.id
            for statement in self.statements
            if statement.subject == statement.id or statement.object == statement.id
        )
        if self_refs:
            raise ValueError(f"golden statements reference themselves: {self_refs}")
        return self

    def terms(self) -> dict[str, ExpectedTerm]:
        return {term.id: term for term in self.entities + self.statements}


class ExtractionCase(BaseModel, extra="forbid"):
    """One article and its expected term graph."""

    article: str
    expected: ExpectedExtraction


def load_dataset() -> Dataset[ExtractionCase, Graph, object]:
    cases = [
        Case[ExtractionCase, Graph, object](
            name=path.stem,
            inputs=ExtractionCase.model_validate(yaml.safe_load(path.read_text())),
        )
        for path in sorted(CASES.glob("*.yaml"))
    ]
    if not cases:
        raise ValueError(f"no extraction eval cases found in {CASES}")
    return Dataset(name="extraction-cases", cases=cases)


# ---------------------------------------------------------------------------
# Task: the production extraction path on one article
# ---------------------------------------------------------------------------


def task(case: ExtractionCase) -> Graph:
    return extraction_to_graph(
        "extraction-eval",
        extract_article(build_agent(MODEL), case.article),
    )


# ---------------------------------------------------------------------------
# Graph comparison
# ---------------------------------------------------------------------------


def render_expected(
    expected: ExpectedExtraction,
    term_id: str,
    stack: frozenset[str] = frozenset(),
) -> str:
    terms = expected.terms()
    term = terms[term_id]
    if isinstance(term, ExpectedEntity):
        return term.name
    if term.id in stack:
        return "⟲"

    inner = frozenset({*stack, term.id})

    def endpoint(endpoint_id: str) -> str:
        rendered = render_expected(expected, endpoint_id, inner)
        return (
            f"[{rendered}]"
            if isinstance(terms[endpoint_id], ExpectedStatement)
            else rendered
        )

    return f"{endpoint(term.subject)} —{term.predicate}→ {endpoint(term.object)}"


def render_term(graph: Graph, term: Term, stack: frozenset[str] = frozenset()) -> str:
    """Render an extracted term; ``stack`` breaks valid reference cycles."""
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


def _compatible(expected: ExpectedTerm, extracted: Term) -> bool:
    if isinstance(expected, ExpectedEntity):
        return isinstance(extracted, Entity) and expected.name in extracted.names
    return (
        isinstance(extracted, Statement)
        and expected.predicate in extracted.predicates
    )


def _extend_mapping(
    expected_terms: dict[str, ExpectedTerm],
    graph: Graph,
    mapping: dict[str, str | None],
    reverse: dict[str, str],
    expected_id: str,
    extracted_id: str,
) -> tuple[dict[str, str | None], dict[str, str]] | None:
    """Map one term and force its endpoint mappings, preserving identity."""
    extended = mapping.copy()
    extended_reverse = reverse.copy()
    pending = [(expected_id, extracted_id)]

    while pending:
        golden_id, runtime_id = pending.pop()
        if golden_id in extended:
            if extended[golden_id] != runtime_id:
                return None
            continue
        if runtime_id in extended_reverse:
            return None

        golden = expected_terms[golden_id]
        extracted = graph.terms[runtime_id]
        if not _compatible(golden, extracted):
            return None

        extended[golden_id] = runtime_id
        extended_reverse[runtime_id] = golden_id
        if isinstance(golden, ExpectedStatement):
            extracted_statement = cast(Statement, extracted)
            pending.extend(
                [
                    (golden.subject, extracted_statement.subject),
                    (golden.object, extracted_statement.object),
                ]
            )

    return extended, extended_reverse


def _best_mapping(graph: Graph, expected: ExpectedExtraction) -> dict[str, str]:
    """Find a maximum structurally consistent injection into ``graph``."""
    expected_terms = expected.terms()
    candidates = {
        golden_id: [
            extracted_id
            for extracted_id, extracted in graph.terms.items()
            if _compatible(golden, extracted)
        ]
        for golden_id, golden in expected_terms.items()
    }
    best: dict[str, str] = {}
    best_score = (-1, -1)

    def search(mapping: dict[str, str | None], reverse: dict[str, str]) -> None:
        nonlocal best, best_score
        mapped = {
            golden_id: runtime_id
            for golden_id, runtime_id in mapping.items()
            if runtime_id is not None
        }
        remaining = len(expected_terms) - len(mapping)
        if len(mapped) + remaining < best_score[0]:
            return
        if not remaining:
            mapped_statements = sum(
                isinstance(expected_terms[golden_id], ExpectedStatement)
                for golden_id in mapped
            )
            score = (len(mapped), mapped_statements)
            if score > best_score:
                best = mapped
                best_score = score
            return

        unassigned = [golden_id for golden_id in expected_terms if golden_id not in mapping]
        golden_id = min(
            unassigned,
            key=lambda term_id: (
                len([candidate for candidate in candidates[term_id] if candidate not in reverse]),
                not isinstance(expected_terms[term_id], ExpectedStatement),
            ),
        )
        for runtime_id in candidates[golden_id]:
            if runtime_id in reverse:
                continue
            extended = _extend_mapping(
                expected_terms,
                graph,
                mapping,
                reverse,
                golden_id,
                runtime_id,
            )
            if extended is not None:
                search(*extended)

        skipped = mapping.copy()
        skipped[golden_id] = None
        search(skipped, reverse)

    search({}, {})
    return best


@dataclass
class Comparison:
    """Precision/recall and both diff directions for one graph mapping."""

    matched_expected_ids: frozenset[str]
    missed_entities: list[str]
    extra_entities: list[str]
    missed_facts: list[str]
    extra_facts: list[str]
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
    mapping = _best_mapping(graph, expected)
    matched_runtime_ids = frozenset(mapping.values())
    matched_expected_ids = frozenset(mapping)
    entities = [term for term in graph.terms.values() if isinstance(term, Entity)]
    statements = [term for term in graph.terms.values() if isinstance(term, Statement)]
    matched_entities = sum(entity.id in matched_runtime_ids for entity in entities)
    matched_statements = sum(statement.id in matched_runtime_ids for statement in statements)

    def ratio(part: int, whole: int) -> float:
        return part / whole if whole else 1.0

    return Comparison(
        matched_expected_ids=matched_expected_ids,
        missed_entities=[
            f"{entity.id}: {entity.name}"
            for entity in expected.entities
            if entity.id not in matched_expected_ids
        ],
        extra_entities=[
            render_term(graph, entity)
            for entity in entities
            if entity.id not in matched_runtime_ids
        ],
        missed_facts=[
            f"{statement.id}: {render_expected(expected, statement.id)}"
            for statement in expected.statements
            if statement.id not in matched_expected_ids
        ],
        extra_facts=[
            render_term(graph, statement)
            for statement in statements
            if statement.id not in matched_runtime_ids
        ],
        entity_precision=ratio(matched_entities, len(entities)),
        entity_recall=ratio(matched_entities, len(expected.entities)),
        fact_precision=ratio(matched_statements, len(statements)),
        fact_recall=ratio(matched_statements, len(expected.statements)),
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class ExtractionHit(Evaluator[ExtractionCase, Graph, object]):
    """One assertion per golden term, plus precision/recall scores."""

    def evaluate(
        self, ctx: EvaluatorContext[ExtractionCase, Graph, object]
    ) -> dict[str, bool | float]:
        expected = ctx.inputs.expected
        result = compare(ctx.output, expected)
        return {
            **{
                f"entity {entity.id}: {entity.name}": (
                    entity.id in result.matched_expected_ids
                )
                for entity in expected.entities
            },
            **{
                f"fact {statement.id}: {render_expected(expected, statement.id)}": (
                    statement.id in result.matched_expected_ids
                )
                for statement in expected.statements
            },
            "entity_precision": round(result.entity_precision, 3),
            "entity_recall": round(result.entity_recall, 3),
            "fact_precision": round(result.fact_precision, 3),
            "fact_recall": round(result.fact_recall, 3),
        }


def main() -> None:
    dataset = load_dataset()
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
        print("  facts    -:", *(result.missed_facts or ["·"]), sep="\n    ")
        print("  facts    +:", *(result.extra_facts or ["·"]), sep="\n    ")


if __name__ == "__main__":
    main()
