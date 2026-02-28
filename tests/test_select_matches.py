"""Unit tests for select_matches (SelectThreshold filter)."""

from worldgraph.match import select_matches


def test_clear_best_match_passes():
    """When one pair dominates from both perspectives, it passes any threshold < 1."""
    sigma = {
        ("a1", "b1"): 0.95,
        ("a1", "b2"): 0.10,
        ("a2", "b1"): 0.05,
        ("a2", "b2"): 0.90,
    }
    matches = select_matches(sigma, ["a1", "a2"], ["b1", "b2"], threshold=0.8)
    assert ("a1", "b1") in matches
    assert ("a2", "b2") in matches
    assert len(matches) == 2


def test_ambiguous_pair_filtered():
    """When a1 scores similarly with b1 and b2 but b1 also has a stronger
    candidate a2, the a1↔b1 pair is filtered from B's side.

    From A's perspective: a1's best is b1 (0.7 vs 0.6), rel_a(a1,b1) = 1.0.
    From B's perspective: b1's best is a2 (0.9), rel_b(a1,b1) = 0.7/0.9 ≈ 0.78.
    At threshold 0.8, a1↔b1 fails the B-side check.
    """
    sigma = {
        ("a1", "b1"): 0.7,
        ("a1", "b2"): 0.6,
        ("a2", "b1"): 0.9,
        ("a2", "b2"): 0.1,
    }
    matches = select_matches(sigma, ["a1", "a2"], ["b1", "b2"], threshold=0.8)
    # a2↔b1 is the strong match (rel_a=1.0, rel_b=1.0)
    assert ("a2", "b1") in matches
    # a1↔b2 is a1's fallback (rel_a=0.6/0.7≈0.86, rel_b=1.0) — passes
    assert ("a1", "b2") in matches
    # a1↔b1 fails B-side: 0.7/0.9 ≈ 0.78 < 0.8
    assert ("a1", "b1") not in matches


def test_below_threshold_from_one_side_rejected():
    """A pair that is the best from A's side but not from B's side is rejected.

    a1's best is b1 (rel_a = 1.0), but b1's best is a2 — so b1→a1 relative
    sim is only 0.6/0.9 ≈ 0.67, below 0.8.
    """
    sigma = {
        ("a1", "b1"): 0.6,
        ("a2", "b1"): 0.9,
    }
    matches = select_matches(sigma, ["a1", "a2"], ["b1"], threshold=0.8)
    # a2↔b1: rel_a = 1.0 (only candidate for a2), rel_b = 1.0 (best for b1)
    assert ("a2", "b1") in matches
    # a1↔b1: rel_a = 1.0 (only candidate for a1), but rel_b = 0.6/0.9 ≈ 0.67
    assert ("a1", "b1") not in matches


def test_all_zero_sigma_returns_empty():
    """When all similarity scores are zero, no matches are returned."""
    sigma = {
        ("a1", "b1"): 0.0,
        ("a1", "b2"): 0.0,
    }
    matches = select_matches(sigma, ["a1"], ["b1", "b2"], threshold=0.5)
    assert matches == []


def test_single_pair_always_matches():
    """With one entity on each side, relative similarity is 1.0 from both
    perspectives — passes any threshold <= 1.0 as long as sigma > 0."""
    sigma = {("a1", "b1"): 0.3}
    matches = select_matches(sigma, ["a1"], ["b1"], threshold=1.0)
    assert matches == [("a1", "b1")]


def test_threshold_boundary():
    """Relative sim exactly at threshold passes; just below does not."""
    # a1→b1: 0.8, a1→b2: 1.0 → rel_a(a1,b1) = 0.8
    # b1 only has a1 → rel_b(a1,b1) = 1.0
    # b2 only has a1 → rel_b(a1,b2) = 1.0
    sigma = {
        ("a1", "b1"): 0.8,
        ("a1", "b2"): 1.0,
    }
    # At threshold=0.8, rel_a for (a1,b1) is exactly 0.8 — should pass
    matches = select_matches(sigma, ["a1"], ["b1", "b2"], threshold=0.8)
    assert ("a1", "b1") in matches
    assert ("a1", "b2") in matches

    # At threshold=0.81, rel_a for (a1,b1) = 0.8 < 0.81 — should fail
    matches = select_matches(sigma, ["a1"], ["b1", "b2"], threshold=0.81)
    assert ("a1", "b1") not in matches
    assert ("a1", "b2") in matches
