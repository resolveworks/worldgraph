"""Layer 1 tests for Soft TF-IDF + Jaro-Winkler name similarity."""

import math

import pytest

from worldgraph.names import build_idf, soft_tfidf


# ---------------------------------------------------------------------------
# build_idf
# ---------------------------------------------------------------------------


def test_rare_token_has_higher_idf_than_common():
    """A token appearing in one label should have higher IDF than one
    appearing in all labels."""
    idf = build_idf(["Apple Inc", "Apple Corp", "DataVault Inc"])
    # "apple" appears in 2/3 labels, "datavault" in 1/3
    assert idf["datavault"] > idf["apple"]


def test_universal_token_has_zero_idf():
    """A token appearing in every label has IDF = log(N/N) = 0."""
    idf = build_idf(["Alpha Corp", "Alpha Inc", "Alpha LLC"])
    assert idf["alpha"] == 0.0


def test_unique_token_has_log_n_idf():
    """A token appearing in exactly one of N labels has IDF = log(N)."""
    idf = build_idf(["Apple", "Google", "Microsoft"])
    assert idf["apple"] == math.log(3)


def test_idf_normalizes_case():
    """Uppercase and lowercase tokens should be treated as the same token."""
    idf = build_idf(["Apple", "APPLE Inc"])
    # "apple" appears in both labels → df=2, IDF = log(2/2) = 0
    assert idf["apple"] == 0.0


# ---------------------------------------------------------------------------
# soft_tfidf — exact and near matches
# ---------------------------------------------------------------------------


def test_identical_strings_return_1():
    idf = build_idf(["Apple", "Google"])
    assert soft_tfidf("Apple", "Apple", idf) == 1.0


def test_identical_multi_token_strings_return_1():
    idf = build_idf(["Meridian Technologies", "DataVault Inc"])
    assert soft_tfidf(
        "Meridian Technologies", "Meridian Technologies", idf
    ) == pytest.approx(1.0)


def test_completely_different_strings_return_0():
    idf = build_idf(["Apple", "Google", "DataVault", "Halcyon"])
    assert soft_tfidf("Apple", "DataVault", idf) == 0.0


def test_abbreviated_name_scores_high():
    """'Meridian Tech' vs 'Meridian Technologies' — 'tech'/'technologies'
    should pass the JW threshold."""
    idf = build_idf(["Meridian Technologies", "Meridian Tech", "DataVault"])
    score = soft_tfidf("Meridian Tech", "Meridian Technologies", idf)
    assert score > 0.8


def test_initial_abbreviation_scores_low():
    """'P. Sharma' vs 'Dr. Priya Sharma' — 'sharma' matches exactly but
    'p.' (high IDF, rare) doesn't match any token above the JW threshold.
    The unmatched high-IDF token dominates the denominator → low score.
    This is a known limitation: initial-only abbreviations need structural
    evidence from propagation to match."""
    labels = ["Dr. Priya Sharma", "P. Sharma", "Elena Vasquez"]
    idf = build_idf(labels)
    score = soft_tfidf("P. Sharma", "Dr. Priya Sharma", idf)
    assert score < 0.5


# ---------------------------------------------------------------------------
# soft_tfidf — no spurious matches
# ---------------------------------------------------------------------------


def test_unrelated_names_score_low():
    idf = build_idf(["Volta Systems", "Halcyon Genomics", "DataVault Inc"])
    score = soft_tfidf("Volta Systems", "Halcyon Genomics", idf)
    assert score < 0.2


def test_different_people_with_shared_title_score_low():
    """'Dr. Priya Sharma' vs 'Dr. Elena Vasquez' — shared 'dr.' is common
    (low IDF), and first/last names are completely different."""
    labels = ["Dr. Priya Sharma", "Dr. Elena Vasquez", "Dr. James Chen"]
    idf = build_idf(labels)
    score = soft_tfidf("Dr. Priya Sharma", "Dr. Elena Vasquez", idf)
    assert score < 0.5


# ---------------------------------------------------------------------------
# JW threshold behavior
# ---------------------------------------------------------------------------


def test_below_jw_threshold_contributes_nothing():
    """Tokens that are close but below the 0.85 JW threshold should
    not contribute to the score."""
    idf = build_idf(["cat", "car"])
    # JW("cat", "car") ≈ 0.82 — below default 0.85 threshold
    assert soft_tfidf("cat", "car", idf) == 0.0


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def test_nfkd_normalization_matches_accented_forms():
    """Accented and non-accented forms of the same name should match."""
    idf = build_idf(["José García", "Jose Garcia", "DataVault"])
    score = soft_tfidf("José García", "Jose Garcia", idf)
    assert score == pytest.approx(1.0)
