"""Entity name similarity via Soft TF-IDF + Jaro-Winkler.

Embedding-based similarity fails for surface-form entity name variations
common in news: abbreviations ("P. Sharma" / "Dr. Priya Sharma"), shortened
names ("Meridian Tech" / "Meridian Technologies"), and acronyms ("FTC" /
"Federal Trade Commission").  Soft TF-IDF with Jaro-Winkler inner similarity
handles these correctly.
"""

import math
import unicodedata
from collections import Counter

from rapidfuzz.distance import JaroWinkler


def _normalize(text: str) -> str:
    """NFKD-normalize and lowercase."""
    return unicodedata.normalize("NFKD", text).lower()


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def build_idf(labels: list[str]) -> dict[str, float]:
    """Compute IDF weights from all entity labels in the corpus.

    IDF(token) = log(N / df(token)) where df is the number of labels
    containing the token.  Tokens are NFKD-normalized and lowercased.
    """
    n = len(labels)
    df: Counter[str] = Counter()
    for label in labels:
        df.update(set(_tokenize(label)))
    return {token: math.log(n / count) for token, count in df.items()}


def soft_tfidf(
    a: str,
    b: str,
    corpus_idf: dict[str, float],
    jw_threshold: float = 0.85,
) -> float:
    """Soft TF-IDF similarity between two label strings.

    For each token in `a`, find the best Jaro-Winkler match in `b`.
    If the best match >= jw_threshold, use the JW score weighted by
    the token's IDF.  Otherwise the token contributes 0.

    Returns a cosine-like score in [0, 1].
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0

    default_idf = max(corpus_idf.values()) if corpus_idf else 1.0

    # Weighted sum for a's tokens matched against b
    numerator = 0.0
    weight_sum_a = 0.0
    for ta in tokens_a:
        idf_a = corpus_idf.get(ta, default_idf)
        weight_sum_a += idf_a**2

        best_jw = 0.0
        for tb in tokens_b:
            jw = JaroWinkler.similarity(ta, tb)
            if jw > best_jw:
                best_jw = jw

        if best_jw >= jw_threshold:
            numerator += idf_a * best_jw * idf_a  # idf² × jw

    # Normalize by the max possible (all tokens matched perfectly)
    if weight_sum_a == 0:
        return 0.0

    return numerator / weight_sum_a
