"""Predicate-similarity prior for statement matching.

Embeds each distinct predicate once at build time and returns a pairwise
callable mapping a statement pair to a prior in [0.5, 1.0]: neutral 0.5
when the embedding signal is absent, rising with cosine similarity above
a baseline, 1.0 for identical predicates.

The prior is a **proposal strength**, not a classification score. The
consumer is a matcher that structurally disposes of false proposals, so
a lifted unrelated pair costs one vetoed proposal, while a missed
paraphrase falls back to neutral-prior structural matching. Separating
paraphrase from unrelated is therefore not a design requirement — the
prior needs only to rank sensibly and stay smooth: no clustering,
thresholds, or equivalence classes.

Measured distribution under the frame below with
sentence-transformers:Qwen/Qwen3-Embedding-0.6B (~30 news predicates,
~330 pairs): paraphrase pairs 0.60–0.93 (median 0.80), unrelated pairs
0.49–0.85 (median 0.68), AUC 0.85. ``_BASELINE_COSINE`` sits just under
the unrelated median: at or below it the embedding carries no usable
signal and the prior stays neutral; above it, cosine is affinely
rescaled from [_BASELINE_COSINE, 1.0] onto [0.5, 1.0] and capped at 1.0.
"""

import os
from collections.abc import Callable

import numpy as np

from worldgraph.embed import Embedder
from worldgraph.graph import Graph, Statement

# Short verb phrases embed unstably in isolation; a minimal frame anchors
# them in a subject-object context. The frame is part of the metric, not
# domain knowledge — filler words must not bias predicates toward a domain.
_FRAME = "A {} B".format

_NEUTRAL_PRIOR = 0.5
_BASELINE_COSINE = 0.65


def _to_prior(cosine: float) -> float:
    if cosine <= _BASELINE_COSINE:
        return _NEUTRAL_PRIOR
    prior = _NEUTRAL_PRIOR + (cosine - _BASELINE_COSINE) * (
        (1.0 - _NEUTRAL_PRIOR) / (1.0 - _BASELINE_COSINE)
    )
    return min(prior, 1.0)


def make_predicate_prior(
    graphs: list[Graph],
    embedder: Embedder | None = None,
) -> Callable[[Statement, Statement], float]:
    """Build a statement-pair prior from predicate embeddings.

    Distinct predicates across *graphs* are embedded once, framed; the
    returned closure only does arithmetic on the cached unit vectors.
    Identical predicate strings short-circuit to 1.0 without a lookup,
    and a predicate unseen at build time scores the neutral 0.5.

    ``embedder=None`` builds the real one from ``EMBEDDING_MODEL``.
    """
    if embedder is None:
        embedder = Embedder(os.environ["EMBEDDING_MODEL"])

    predicates = sorted(
        {
            term.predicate
            for graph in graphs
            for term in graph.terms.values()
            if isinstance(term, Statement)
        }
    )
    vectors = embedder.embed(predicates, template=_FRAME)

    def prior(a: Statement, b: Statement) -> float:
        if a.predicate == b.predicate:
            return 1.0
        va = vectors.get(a.predicate)
        vb = vectors.get(b.predicate)
        if va is None or vb is None:
            return _NEUTRAL_PRIOR
        return _to_prior(float(np.dot(va, vb)))

    return prior
