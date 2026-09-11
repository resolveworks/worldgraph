"""Cross-article merge eval: pydantic-evals dataset runner over article pairs.

Cases in ``datasets/merges.yaml`` pair two fixture articles and state merge
expectations against the matcher's union-find output. A term is located in
its article's graph by surface form — an entity by name, a statement by
subject name, predicate, and object name, nested the same way when an
endpoint is itself a statement. The suite judges the product, not the
extraction: an expectation whose term cannot be located fails its
assertion with an ``extraction miss`` reason — the miss is a finding, not
a crash — and matching failures carry the groups the terms landed in.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from pydantic import BaseModel
from pydantic_evals import Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from worldgraph.embed import Embedder
from worldgraph.extract import build_agent, extract_article, extraction_to_graph
from worldgraph.graph import Entity, Graph, Statement, Term
from worldgraph.match import MatchGroup, Qid, match_graphs
from worldgraph.priors import make_predicate_prior

FIXTURES = Path(__file__).parent / "fixtures"
DATASET_PATH = Path(__file__).parent / "datasets" / "merges.yaml"

MODEL = os.environ["EXTRACTION_MODEL"]


# ---------------------------------------------------------------------------
# Case schema
# ---------------------------------------------------------------------------


class FactRef(BaseModel, extra="forbid"):
    """A statement located by surface form: ``predicate`` plus each
    endpoint's surface form — an entity name, or the referenced
    statement's own ``FactRef`` when the endpoint nests."""

    subject: str | FactRef
    predicate: str
    object: str | FactRef


Ref = str | FactRef


class Expectation(BaseModel, extra="forbid"):
    """A pair of terms — ``a`` in article A, ``b`` in article B — that
    must (or must not) share a union-find group after matching."""

    a: Ref
    b: Ref


class MergeCase(BaseModel, extra="forbid"):
    article_a: str  # fixture stem
    article_b: str  # fixture stem
    expected_merges: list[Expectation]
    expected_non_merges: list[Expectation]


# ---------------------------------------------------------------------------
# Task: the production pipeline over an article pair
# ---------------------------------------------------------------------------


@dataclass
class MatchOutput:
    graphs: dict[str, Graph]  # fixture stem → extracted graph
    groups: list[MatchGroup]  # union-find groups from the matcher


def run_case(case: MergeCase, embedder: Embedder) -> MatchOutput:
    """Extract both articles and match them — the production path end to
    end, with the real predicate prior."""
    agent = build_agent(MODEL)
    graphs = {
        stem: extraction_to_graph(
            stem, extract_article(agent, (FIXTURES / f"{stem}.md").read_text())
        )
        for stem in (case.article_a, case.article_b)
    }
    prior = make_predicate_prior(list(graphs.values()), embedder=embedder)
    _confidence, groups, _merged = match_graphs(
        list(graphs.values()), predicate_prior=prior
    )
    return MatchOutput(graphs=graphs, groups=groups)


# ---------------------------------------------------------------------------
# Locating terms by surface form
# ---------------------------------------------------------------------------


def _matches(graph: Graph, term: Term, ref: Ref) -> bool:
    if isinstance(ref, str):
        return isinstance(term, Entity) and ref in term.names
    return (
        isinstance(term, Statement)
        and term.predicate == ref.predicate
        and _matches(graph, graph.terms[term.subject], ref.subject)
        and _matches(graph, graph.terms[term.object], ref.object)
    )


def locate(graph: Graph, ref: Ref) -> list[Term]:
    """All terms of *graph* whose surface form matches *ref*."""
    return [term for term in graph.terms.values() if _matches(graph, term, ref)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(ref: Ref) -> str:
    """Compact surface form: names plain, statements as
    ``subject —predicate→ object`` with nested statements bracketed."""
    if isinstance(ref, str):
        return ref

    def endpoint(e: Ref) -> str:
        return e if isinstance(e, str) else f"[{render(e)}]"

    return f"{endpoint(ref.subject)} —{ref.predicate}→ {endpoint(ref.object)}"


def render_term(graph: Graph, term: Term, stack: frozenset[str] = frozenset()) -> str:
    """Render an extracted term the same way; ``stack`` breaks reference
    cycles, which valid graphs may contain."""
    if isinstance(term, Entity):
        return term.names[0]
    if term.id in stack:
        return "⟲"
    inner = frozenset({*stack, term.id})
    return (
        f"{render_term(graph, graph.terms[term.subject], inner)}"
        f" —{term.predicate}→ "
        f"{render_term(graph, graph.terms[term.object], inner)}"
    )


# ---------------------------------------------------------------------------
# Evaluator: one assertion per expectation
# ---------------------------------------------------------------------------


def _describe(out: MatchOutput, group_of: dict[Qid, MatchGroup], qid: Qid) -> str:
    """Where one located term ended up: its label, article, and group."""
    graph = out.graphs[qid[0]]
    term = graph.terms[qid[1]]
    where = f"{render_term(graph, term)} ({qid[0]})"
    group = group_of.get(qid)
    if group is None:
        return f"{where} is a singleton"
    peers = ", ".join(
        sorted(
            render_term(out.graphs[m[0]], out.graphs[m[0]].terms[m[1]])
            for m in group
            if m != qid
        )
    )
    return f"{where} groups with {peers}"


@dataclass
class MergeExpectations(Evaluator[MergeCase, MatchOutput, object]):
    """Turn each expectation into one named assertion.

    A merge passes when some located instance of ``a`` shares a group
    with some located instance of ``b``; a non-merge passes when no
    located pair shares one. An unlocatable ref fails with an
    ``extraction miss`` reason — a distinct, reportable outcome.
    """

    def evaluate(
        self, ctx: EvaluatorContext[MergeCase, MatchOutput, object]
    ) -> dict[str, EvaluationReason]:
        out = ctx.output
        case = ctx.inputs
        group_of: dict[Qid, MatchGroup] = {
            member: group for group in out.groups for member in group
        }

        def judge(exp: Expectation, *, must_share: bool) -> EvaluationReason:
            graph_a = out.graphs[case.article_a]
            graph_b = out.graphs[case.article_b]
            found_a = [(case.article_a, term.id) for term in locate(graph_a, exp.a)]
            found_b = [(case.article_b, term.id) for term in locate(graph_b, exp.b)]
            if not found_a:
                return EvaluationReason(
                    value=False,
                    reason=f"extraction miss: {render(exp.a)} not found"
                    f" in {case.article_a}",
                )
            if not found_b:
                return EvaluationReason(
                    value=False,
                    reason=f"extraction miss: {render(exp.b)} not found"
                    f" in {case.article_b}",
                )
            sharing = [
                (qa, qb)
                for qa in found_a
                for qb in found_b
                if (group := group_of.get(qa)) is not None
                and group is group_of.get(qb)
            ]
            if bool(sharing) == must_share:
                return EvaluationReason(value=True)
            verb = "never merged" if must_share else "merged"
            detail = "; ".join(_describe(out, group_of, q) for q in found_a + found_b)
            return EvaluationReason(
                value=False,
                reason=f"{render(exp.a)} and {render(exp.b)} {verb} — {detail}",
            )

        assertions = {
            f"merge: {render(exp.a)} ↔ {render(exp.b)}": judge(exp, must_share=True)
            for exp in case.expected_merges
        }
        assertions.update(
            {
                f"distinct: {render(exp.a)} ≠ {render(exp.b)}": judge(
                    exp, must_share=False
                )
                for exp in case.expected_non_merges
            }
        )
        return assertions


def main() -> None:
    # One embedder and sequential cases: concurrent loads of the embedding
    # model in one process are not thread-safe.
    task: Callable[[MergeCase], MatchOutput] = partial(
        run_case, embedder=Embedder(os.environ["EMBEDDING_MODEL"])
    )
    dataset = Dataset[MergeCase, MatchOutput, object].from_file(DATASET_PATH)
    dataset.evaluators.append(MergeExpectations())
    report = dataset.evaluate_sync(task, max_concurrency=1)
    report.print(include_reasons=True)


if __name__ == "__main__":
    main()
