"""Unit tests for select_matches (confidence threshold filter)."""

from worldgraph.match import select_matches


def test_above_threshold_matches():
    """A pair passes when confidence exceeds the threshold."""
    confidence = {("a1", "b1"): 0.9}
    matches = select_matches(confidence, ["a1"], ["b1"], threshold=0.8)
    assert matches == [("a1", "b1")]


def test_below_threshold_rejected():
    """A pair with confidence below the threshold is rejected."""
    confidence = {("a1", "b1"): 0.5}
    matches = select_matches(confidence, ["a1"], ["b1"], threshold=0.8)
    assert matches == []


def test_threshold_boundary():
    """Score exactly at threshold passes; just below does not."""
    confidence = {("a1", "b1"): 0.8}
    matches = select_matches(confidence, ["a1"], ["b1"], threshold=0.8)
    assert matches == [("a1", "b1")]

    matches = select_matches(confidence, ["a1"], ["b1"], threshold=0.81)
    assert matches == []


def test_multiple_pairs_filtered_independently():
    """Each pair is evaluated independently against the threshold."""
    confidence = {
        ("a1", "b1"): 0.95,
        ("a1", "b2"): 0.3,
        ("a2", "b1"): 0.4,
        ("a2", "b2"): 0.9,
    }
    matches = select_matches(
        confidence, ["a1", "a2"], ["b1", "b2"], threshold=0.8
    )
    assert ("a1", "b1") in matches
    assert ("a2", "b2") in matches
    assert ("a1", "b2") not in matches
    assert ("a2", "b1") not in matches
    assert len(matches) == 2


def test_missing_pair_treated_as_zero():
    """A pair not present in the confidence dict is treated as zero."""
    confidence = {}
    matches = select_matches(confidence, ["a1"], ["b1"], threshold=0.8)
    assert matches == []
