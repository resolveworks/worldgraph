"""Extraction eval harness: pydantic-evals dataset runner over article fixtures.

Cases in ``datasets/extraction.yaml`` name fixture stems under ``fixtures/``.
The task runs the production extraction path on each article; the
``ExtractionPRF`` evaluator scores the predicted extraction against the
hand-labeled expected output with tolerant entity and relation matching.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from worldgraph.constants import (
    MERGE_THRESHOLD,
    RELATION_TEMPLATE,
    RELATION_THRESHOLD,
)
from worldgraph.embed import Embedder
from worldgraph.extract import Extraction, build_agent, extract_article
from worldgraph.names import build_idf, soft_tfidf

load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"
DATASET_PATH = Path(__file__).parent / "datasets" / "extraction.yaml"

MODEL = "deepseek:deepseek-v4-flash"  # keep in sync with worldgraph/cli.py default


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def task(stem: str) -> Extraction:
    """Run the production extraction path on a fixture article."""
    return extract_article(build_agent(MODEL), (FIXTURES / f"{stem}.md").read_text())


# ---------------------------------------------------------------------------
# Matching and evaluation
# ---------------------------------------------------------------------------


def match_entities(pred: Extraction, gold: Extraction) -> dict[int, int]:
    """Greedy one-to-one matching of predicted to golden entity indices.

    A pair matches iff its Soft TF-IDF name similarity reaches the
    pipeline's merge threshold (the same cutoff that gates name-seeded
    merges in ``worldgraph.match``).
    """
    idf = build_idf([e.name for e in pred.entities] + [e.name for e in gold.entities])
    candidates = [
        (soft_tfidf(p.name, g.name, idf), i, j)
        for i, p in enumerate(pred.entities)
        for j, g in enumerate(gold.entities)
    ]
    matched: dict[int, int] = {}
    taken_gold: set[int] = set()
    for sim, i, j in sorted(candidates, reverse=True):
        if sim < MERGE_THRESHOLD:
            break
        if i in matched or j in taken_gold:
            continue
        matched[i] = j
        taken_gold.add(j)
    return matched


def _prf(tp: int, pred_n: int, gold_n: int) -> tuple[float, float, float]:
    """Precision/recall/F1; an empty side is a perfect score for it."""
    precision = tp / pred_n if pred_n else 1.0
    recall = tp / gold_n if gold_n else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def extraction_scores(
    pred: Extraction, gold: Extraction, embedder: Embedder
) -> dict[str, float]:
    """Precision/recall/F1 of entities and edges with tolerant matching.

    Entities match via greedy Soft TF-IDF name matching. An edge matches
    iff its endpoints are matched and the relation phrases are
    cosine-similar above the relation equivalence threshold.
    """
    matched = match_entities(pred, gold)

    pred_edges = [
        (
            next(i for i, e in enumerate(pred.entities) if e.id == r.source),
            next(i for i, e in enumerate(pred.entities) if e.id == r.target),
            r.relation,
        )
        for r in pred.relations
    ]
    gold_edges = [
        (
            next(j for j, e in enumerate(gold.entities) if e.id == r.source),
            next(j for j, e in enumerate(gold.entities) if e.id == r.target),
            r.relation,
        )
        for r in gold.relations
    ]

    relations = sorted({r for _, _, r in pred_edges + gold_edges})
    emb = embedder.embed(relations, template=RELATION_TEMPLATE)

    inv = {v: k for k, v in matched.items()}
    matched_pred_edges = 0
    matched_gold_edges = 0
    for pi, pj, pr in pred_edges:
        if pi not in matched or pj not in matched:
            continue
        gi, gj = matched[pi], matched[pj]
        if any(
            ni == gi
            and nj == gj
            and float(np.dot(emb[pr], emb[gr])) >= RELATION_THRESHOLD
            for ni, nj, gr in gold_edges
        ):
            matched_pred_edges += 1
    for gi, gj, gr in gold_edges:
        if gi not in inv or gj not in inv:
            continue
        pi, pj = inv[gi], inv[gj]
        if any(
            ni == pi
            and nj == pj
            and float(np.dot(emb[pr], emb[gr])) >= RELATION_THRESHOLD
            for ni, nj, pr in pred_edges
        ):
            matched_gold_edges += 1

    e_p, e_r, e_f1 = _prf(len(matched), len(pred.entities), len(gold.entities))
    d_p, d_r, d_f1 = _prf(matched_pred_edges, len(pred_edges), len(gold_edges))
    return {
        "entity_precision": e_p,
        "entity_recall": e_r,
        "entity_f1": e_f1,
        "edge_precision": d_p,
        "edge_recall": d_r,
        "edge_f1": d_f1,
    }


@dataclass
class ExtractionPRF(Evaluator[Extraction, Extraction, object]):
    """Reference-based precision/recall of an extraction against golden data."""

    embedder: Embedder

    def evaluate(self, ctx: EvaluatorContext[Extraction, Extraction, object]):
        assert ctx.expected_output is not None
        return extraction_scores(ctx.output, ctx.expected_output, self.embedder)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    dataset: Dataset[str, Extraction, object] = Dataset.from_file(DATASET_PATH)
    dataset.evaluators.append(
        ExtractionPRF(Embedder(os.environ["EMBEDDING_MODEL"]))
    )
    report = dataset.evaluate_sync(task)
    report.print(include_reasons=True, include_averages=True)


if __name__ == "__main__":
    main()
