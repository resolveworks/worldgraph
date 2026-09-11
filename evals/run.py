"""Extraction eval harness over self-contained YAML cases.

Each file under ``cases/`` contains an article and its expected term graph;
the filename is the case name. The golden is a ``Graph`` whose local ids
express term identity and shared references only: comparison finds the
largest structurally consistent mapping to the extracted graph, so ids are
never compared. The evaluator scores entity and statement precision/recall
and prints misses and extras in both directions. The matcher is deterministic
and covered by unit tests — this suite judges only extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel
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


class ExpectedExtraction(BaseModel, extra="forbid"):
    """A golden term graph; ids are local to the case."""

    entities: list[ExpectedEntity]
    statements: list[ExpectedStatement]

    def to_graph(self) -> Graph:
        graph = Graph(id="golden")
        for entity in self.entities:
            graph.add_entity(entity.name, id=entity.id)
        for statement in self.statements:
            graph.add_statement(
                statement.subject,
                statement.predicate,
                statement.object,
                id=statement.id,
            )
        graph.validate()
        return graph


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


def render_term(graph: Graph, term: Term, stack: frozenset[str] = frozenset()) -> str:
    """Render a term; ``stack`` breaks reference cycles."""
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


def _compatible(golden: Term, extracted: Term) -> bool:
    if isinstance(golden, Entity):
        return isinstance(extracted, Entity) and golden.names[0] in extracted.names
    return isinstance(extracted, Statement) and golden.predicates[0] in extracted.predicates


def _extend_mapping(
    golden: Graph,
    graph: Graph,
    mapping: dict[str, str | None],
    reverse: dict[str, str],
    golden_id: str,
    extracted_id: str,
) -> tuple[dict[str, str | None], dict[str, str]] | None:
    """Map one golden term, forcing its endpoint mappings; None on conflict."""
    extended = mapping.copy()
    extended_reverse = reverse.copy()
    pending = [(golden_id, extracted_id)]

    while pending:
        gid, rid = pending.pop()
        if gid in extended:
            if extended[gid] != rid:
                return None
            continue
        if rid in extended_reverse:
            return None

        golden_term = golden.terms[gid]
        extracted = graph.terms[rid]
        if not _compatible(golden_term, extracted):
            return None

        extended[gid] = rid
        extended_reverse[rid] = gid
        if isinstance(golden_term, Statement):
            assert isinstance(extracted, Statement)
            pending.append((golden_term.subject, extracted.subject))
            pending.append((golden_term.object, extracted.object))

    return extended, extended_reverse


def _best_mapping(graph: Graph, golden: Graph) -> dict[str, str]:
    """Find a maximum structurally consistent injection of ``golden`` into
    ``graph``, preferring statement matches on ties."""
    candidates = {
        golden_id: [
            extracted_id
            for extracted_id, extracted in graph.terms.items()
            if _compatible(golden_term, extracted)
        ]
        for golden_id, golden_term in golden.terms.items()
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
        if len(mapped) + len(golden.terms) - len(mapping) < best_score[0]:
            return
        if len(mapping) == len(golden.terms):
            mapped_statements = sum(
                isinstance(golden.terms[golden_id], Statement) for golden_id in mapped
            )
            score = (len(mapped), mapped_statements)
            if score > best_score:
                best = mapped
                best_score = score
            return

        golden_id = min(
            (gid for gid in golden.terms if gid not in mapping),
            key=lambda gid: (
                len([c for c in candidates[gid] if c not in reverse]),
                not isinstance(golden.terms[gid], Statement),
            ),
        )
        for extracted_id in candidates[golden_id]:
            if extracted_id in reverse:
                continue
            extended = _extend_mapping(
                golden, graph, mapping, reverse, golden_id, extracted_id
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

    matched_golden_ids: frozenset[str]
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


def compare(graph: Graph, golden: Graph) -> Comparison:
    mapping = _best_mapping(graph, golden)
    matched_runtime_ids = frozenset(mapping.values())
    matched_golden_ids = frozenset(mapping)
    entities = [term for term in graph.terms.values() if isinstance(term, Entity)]
    statements = [term for term in graph.terms.values() if isinstance(term, Statement)]
    golden_entities = [
        term for term in golden.terms.values() if isinstance(term, Entity)
    ]
    golden_statements = [
        term for term in golden.terms.values() if isinstance(term, Statement)
    ]
    matched_entities = sum(entity.id in matched_runtime_ids for entity in entities)
    matched_statements = sum(statement.id in matched_runtime_ids for statement in statements)

    def ratio(part: int, whole: int) -> float:
        return part / whole if whole else 1.0

    return Comparison(
        matched_golden_ids=matched_golden_ids,
        missed_entities=[
            f"{term.id}: {term.names[0]}"
            for term in golden_entities
            if term.id not in matched_golden_ids
        ],
        extra_entities=[
            render_term(graph, entity)
            for entity in entities
            if entity.id not in matched_runtime_ids
        ],
        missed_facts=[
            f"{term.id}: {render_term(golden, term)}"
            for term in golden_statements
            if term.id not in matched_golden_ids
        ],
        extra_facts=[
            render_term(graph, statement)
            for statement in statements
            if statement.id not in matched_runtime_ids
        ],
        entity_precision=ratio(matched_entities, len(entities)),
        entity_recall=ratio(matched_entities, len(golden_entities)),
        fact_precision=ratio(matched_statements, len(statements)),
        fact_recall=ratio(matched_statements, len(golden_statements)),
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
        golden = ctx.inputs.expected.to_graph()
        result = compare(ctx.output, golden)
        return {
            **{
                f"entity {term.id}: {term.names[0]}": term.id in result.matched_golden_ids
                for term in golden.terms.values()
                if isinstance(term, Entity)
            },
            **{
                f"fact {term.id}: {render_term(golden, term)}": (
                    term.id in result.matched_golden_ids
                )
                for term in golden.terms.values()
                if isinstance(term, Statement)
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
        golden = case.inputs.expected.to_graph()
        result = compare(case.output, golden)
        if result.clean():
            continue
        print(f"\n{case.name}:")
        print("  entities -:", *(result.missed_entities or ["·"]), sep="\n    ")
        print("  entities +:", *(result.extra_entities or ["·"]), sep="\n    ")
        print("  facts    -:", *(result.missed_facts or ["·"]), sep="\n    ")
        print("  facts    +:", *(result.extra_facts or ["·"]), sep="\n    ")


if __name__ == "__main__":
    main()
