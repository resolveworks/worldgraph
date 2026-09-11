"""Predicate-similarity prior for statement matching.

Embeds each distinct predicate once at build time and returns a pairwise
callable mapping a statement pair to a prior in [0.5, 1.0]: neutral 0.5
when the embedding signal is absent, rising with cosine similarity above
a baseline, 1.0 for identical predicates. The prior proposes only —
structural matching disposes — so it is a smooth score: no clustering,
thresholds, or equivalence classes.

Constants were calibrated on a 30-predicate news-relation set (paraphrase
groups vs unrelated groups, ~330 pairs) embedded with
sentence-transformers:Qwen/Qwen3-Embedding-0.6B under the frame below:
paraphrase pairs score >= 0.81, unrelated pairs median 0.74 (p90 0.82).

- ``_FRAME``: short verb phrases embed unstably in isolation; generic
  company/startup filler grounds the phrase in a news-sentence reading.
  The old two-letter frame ("A {} B") left the distributions overlapping
  (AUC 0.85 vs 0.99 for the sentence frame) because Qwen3-Embedding
  compresses cosines on near-skeletal strings.
- ``_BASELINE_COSINE`` = 0.75: the cosine at or below which the embedding
  signal is treated as absent — above the unrelated-pair mass, below the
  paraphrase floor. Above it, cosine is affinely rescaled from
  [_BASELINE_COSINE, 1.0] onto [0.5, 1.0] and capped at 1.0.
"""

import os
from collections.abc import Callable

import numpy as np

from worldgraph.embed import Embedder
from worldgraph.graph import Graph, Statement

# Short verb phrases embed unstably in isolation; generic company/startup
# filler grounds them in a news-sentence reading (see module docstring).
_FRAME = "The company {} the startup".format

_NEUTRAL_PRIOR = 0.5
_BASELINE_COSINE = 0.75


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
