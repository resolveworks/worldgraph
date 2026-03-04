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
    """NFKD-normalize, strip combining marks, and lowercase."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


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

    Following Cohen, Ravikumar & Fienberg (2003):

        SoftTFIDF(S, T) = Σ_{w ∈ CLOSE(θ,S,T)} V(w,S) · V(w',T) · D(w,T)

    where V(w, X) is the L2-normalized IDF weight of token w in string X,
    D(w, T) is the best Jaro-Winkler match for w in T, and CLOSE is the
    set of tokens in S with a JW match >= θ in T.

    We use θ=0.85 (paper uses 0.9) to capture prefix-truncation
    variations common in news entity names (tech/technologies ≈ 0.87).

    Returns score in [0, 1].
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0

    default_idf = max(corpus_idf.values()) if corpus_idf else 1.0

    def _idf(token: str) -> float:
        return corpus_idf.get(token, default_idf)

    # L2 norms for normalization
    norm_a = math.sqrt(sum(_idf(t) ** 2 for t in tokens_a))
    norm_b = math.sqrt(sum(_idf(t) ** 2 for t in tokens_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Σ V(w, S) · V(w', T) · D(w, T) for w ∈ CLOSE(θ, S, T)
    score = 0.0
    for ta in tokens_a:
        best_jw = 0.0
        best_tb = None
        for tb in tokens_b:
            jw = JaroWinkler.similarity(ta, tb)
            if jw > best_jw:
                best_jw = jw
                best_tb = tb

        if best_jw >= jw_threshold and best_tb is not None:
            v_a = _idf(ta) / norm_a
            v_b = _idf(best_tb) / norm_b
            score += v_a * v_b * best_jw

    return score
