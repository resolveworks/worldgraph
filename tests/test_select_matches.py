"""Unit tests for select_matches (absolute threshold filter)."""

from worldgraph.match import select_matches


def test_both_above_threshold_matches():
    """A pair passes when both name_sim and structural exceed their thresholds."""
    name_sim = {("a1", "b1"): 0.9}
    structural = {("a1", "b1"): 0.85}
    matches = select_matches(
        name_sim,
        structural,
        ["a1"],
        ["b1"],
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == [("a1", "b1")]


def test_low_name_sim_rejected():
    """A pair with high structural but low name similarity is rejected."""
    name_sim = {("a1", "b1"): 0.5}
    structural = {("a1", "b1"): 0.95}
    matches = select_matches(
        name_sim,
        structural,
        ["a1"],
        ["b1"],
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == []


def test_low_structural_rejected():
    """A pair with high name similarity but low structural evidence is rejected."""
    name_sim = {("a1", "b1"): 0.95}
    structural = {("a1", "b1"): 0.1}
    matches = select_matches(
        name_sim,
        structural,
        ["a1"],
        ["b1"],
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == []


def test_zero_structural_rejected():
    """A pair with no structural evidence is rejected even with perfect name match."""
    name_sim = {("a1", "b1"): 1.0}
    structural = {("a1", "b1"): 0.0}
    matches = select_matches(
        name_sim,
        structural,
        ["a1"],
        ["b1"],
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == []


def test_threshold_boundary():
    """Scores exactly at threshold pass; just below do not."""
    name_sim = {("a1", "b1"): 0.8}
    structural = {("a1", "b1"): 0.8}
    matches = select_matches(
        name_sim,
        structural,
        ["a1"],
        ["b1"],
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == [("a1", "b1")]

    matches = select_matches(
        name_sim,
        structural,
        ["a1"],
        ["b1"],
        name_threshold=0.81,
        structural_threshold=0.8,
    )
    assert matches == []


def test_multiple_pairs_filtered_independently():
    """Each pair is evaluated independently against the thresholds."""
    name_sim = {
        ("a1", "b1"): 0.95,
        ("a1", "b2"): 0.3,
        ("a2", "b1"): 0.4,
        ("a2", "b2"): 0.9,
    }
    structural = {
        ("a1", "b1"): 0.9,
        ("a1", "b2"): 0.9,
        ("a2", "b1"): 0.9,
        ("a2", "b2"): 0.85,
    }
    matches = select_matches(
        name_sim,
        structural,
        ["a1", "a2"],
        ["b1", "b2"],
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert ("a1", "b1") in matches
    assert ("a2", "b2") in matches
    assert ("a1", "b2") not in matches
    assert ("a2", "b1") not in matches
    assert len(matches) == 2
