"""Layer 2 tests for similarity propagation over entity-event graphs.

Graphs are bipartite: entity nodes connect to event nodes via role edges
(agent/patient). Entity pairs are seeded by name similarity; event pairs
start at the neutral prior (0.5) and are lifted or suppressed purely by
structural evidence. Tests verify that:

- Matching names + matching structure produce correct matches
- Synonym event labels are irrelevant: structure alone merges events
- A single shared participant never merges an event pair, however the
  labels read — the merge bar requires corroborating paths
- Mismatched participants suppress event pairs below the prior (negative
  evidence), and through them the participating entities
- Evidence propagates to both agent-side and patient-side participants
- Functionality weighting affects evidence strength
- Multi-hop chains require iterative propagation
- Name variation with structural reinforcement (the core use case)
- Dangling entities get no structural evidence
- Multiple matched events accumulate evidence
"""

import pytest
from conftest import fact

from worldgraph.graph import Graph
from worldgraph.match import match_graphs
from worldgraph.names import build_idf, soft_tfidf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_matches(confidence, threshold=0.8):
    """Select matches: pairs where confidence >= threshold."""
    seen = set()
    matches = []
    for (id_a, id_b), score in confidence.items():
        if score >= threshold and (id_b, id_a) not in seen:
            matches.append((id_a, id_b))
            seen.add((id_a, id_b))
    return matches


# ---------------------------------------------------------------------------
# Correct matches
# ---------------------------------------------------------------------------


def test_matching_names_and_structure_produce_matches():
    """Two graphs with the same entity names and the same event structure
    produce high-confidence matches for the correct entity pairs."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    fact(g1, "acquire", agent=apple1, patients=(beats1,))

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    fact(g2, "acquire", agent=apple2, patients=(beats2,))

    confidence, _, _ = match_graphs([g1, g2])

    assert confidence[(apple1.id, apple2.id)] > 0.8
    assert confidence[(beats1.id, beats2.id)] > 0.8

    assert confidence[(apple1.id, beats2.id)] < 0.5
    assert confidence[(beats1.id, apple2.id)] < 0.5


# ---------------------------------------------------------------------------
# No spurious matches
# ---------------------------------------------------------------------------


def test_unrelated_graphs_do_not_match():
    """Entirely disjoint entities and events: no matches."""
    g1 = Graph(id="g1")
    apple = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    fact(g1, "acquire", agent=apple, patients=(beats,))

    g2 = Graph(id="g2")
    tokyo = g2.add_entity("Tokyo")
    japan = g2.add_entity("Japan")
    fact(g2, "be located in", agent=tokyo, patients=(japan,))

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    assert matches == [], f"Spurious matches found: {matches}"


def test_weak_neighbors_do_not_produce_matches():
    """Identical event shape but dissimilar participant names on both
    sides: negative evidence suppresses the event pair, nothing merges."""
    g1 = Graph(id="g1")
    apple = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    fact(g1, "acquire", agent=apple, patients=(beats,))

    g2 = Graph(id="g2")
    google = g2.add_entity("Google")
    youtube = g2.add_entity("YouTube")
    fact(g2, "acquire", agent=google, patients=(youtube,))

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    assert matches == [], f"Spurious matches from weak neighbors: {matches}"


def test_many_weak_paths_do_not_accumulate():
    """Many unrelated events should not produce matches."""
    g1 = Graph(id="g1")
    org = g1.add_entity("Org")
    target = g1.add_entity("Target")
    project = g1.add_entity("Project")
    person = g1.add_entity("Person")
    fact(g1, "acquire", agent=org, patients=(target,))
    fact(g1, "fund", agent=org, patients=(project,))
    fact(g1, "hire", agent=org, patients=(person,))

    g2 = Graph(id="g2")
    city = g2.add_entity("City")
    country = g2.add_entity("Country")
    river = g2.add_entity("River")
    venue = g2.add_entity("Venue")
    fact(g2, "be located in", agent=city, patients=(country,))
    fact(g2, "border", agent=city, patients=(river,))
    fact(g2, "host", agent=city, patients=(venue,))

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    assert matches == [], f"Spurious matches from accumulated weak paths: {matches}"


# ---------------------------------------------------------------------------
# The merge bar: corroborating paths
# ---------------------------------------------------------------------------


def test_single_shared_participant_does_not_merge_events():
    """'John resigned' vs 'John purchased a car': one role-aligned path
    (the shared agent) is never enough to merge an event pair."""
    g1 = Graph(id="g1")
    john1 = g1.add_entity("John")
    resignation = fact(g1, "resign", agent=john1)

    g2 = Graph(id="g2")
    john2 = g2.add_entity("John")
    car2 = g2.add_entity("car")
    purchase = fact(g2, "purchase", agent=john2, patients=(car2,))

    _, groups, _ = match_graphs([g1, g2])

    for group in groups:
        assert not {resignation.id, purchase.id} <= group, (
            "single-path event pair merged"
        )


def test_single_path_events_do_not_merge_even_with_identical_labels():
    """Two outlets both report 'John resigned' with no further shared
    detail. Labels are never compared, so this is a single-path pair —
    and single-path candidates do not merge. Confirmation requires
    corroborating structure, not wording."""
    g1 = Graph(id="g1")
    john1 = g1.add_entity("John")
    resignation1 = fact(g1, "resign", agent=john1)

    g2 = Graph(id="g2")
    john2 = g2.add_entity("John")
    resignation2 = fact(g2, "resign", agent=john2)

    _, groups, _ = match_graphs([g1, g2])

    for group in groups:
        assert not {resignation1.id, resignation2.id} <= group, (
            "single-path event pair merged on label identity"
        )


def test_mismatched_participant_suppresses_event_and_participants():
    """'John purchased a car' vs 'John purchased a truck': the matched
    agent path is outweighed by the mismatched patient path — the event
    pair is suppressed below the prior, and the same-name Johns (whose
    only structural interaction is the suppressed event) do not merge."""
    g1 = Graph(id="g1")
    john1 = g1.add_entity("John")
    car1 = g1.add_entity("car")
    purchase1 = fact(g1, "purchase", agent=john1, patients=(car1,))

    g2 = Graph(id="g2")
    john2 = g2.add_entity("John")
    truck2 = g2.add_entity("truck")
    purchase2 = fact(g2, "purchase", agent=john2, patients=(truck2,))

    confidence, groups, _ = match_graphs([g1, g2])

    assert confidence[(purchase1.id, purchase2.id)] < 0.5, (
        f"mismatched patient did not suppress event pair: "
        f"{confidence[(purchase1.id, purchase2.id)]:.3f}"
    )
    for group in groups:
        assert not {purchase1.id, purchase2.id} <= group
        assert not {john1.id, john2.id} <= group, (
            "johns merged despite their only shared event being suppressed"
        )


# ---------------------------------------------------------------------------
# Evidence direction
# ---------------------------------------------------------------------------


def test_agent_side_evidence_propagates():
    """Evidence reaches participants on the agent side of events.

    The observed pair (Axiom Corp / Pinnacle Ltd) are *agents* with zero
    name similarity; the identical-name patient (DataVault) anchors the
    event pair, whose lift then propagates to the agents."""
    g1 = Graph(id="g1")
    src1 = g1.add_entity("Axiom Corp")
    tv1 = g1.add_entity("DataVault")
    fact(g1, "acquire", agent=src1, patients=(tv1,))

    g2 = Graph(id="g2")
    src2 = g2.add_entity("Pinnacle Ltd")
    tv2 = g2.add_entity("DataVault")
    fact(g2, "acquire", agent=src2, patients=(tv2,))

    # Premise: the observed pair has no name signal; the anchor is maximal
    idf = build_idf(["Axiom Corp", "Pinnacle Ltd", "DataVault"])
    assert soft_tfidf("Axiom Corp", "Pinnacle Ltd", idf) < 0.1
    assert soft_tfidf("DataVault", "DataVault", idf) == 1.0

    confidence, _, _ = match_graphs([g1, g2])

    assert confidence[(src1.id, src2.id)] > 0, (
        "agent-side path did not propagate anchor confidence"
    )


def test_patient_side_evidence_propagates():
    """Evidence reaches participants on the patient side of events.

    Mirror of test_agent_side_evidence_propagates: the observed pair
    (VaultWorks / CloudScale) are *patients* with zero name similarity;
    the identical-name agent (Axiom Corp) anchors."""
    g1 = Graph(id="g1")
    src1 = g1.add_entity("Axiom Corp")
    tv1 = g1.add_entity("VaultWorks")
    fact(g1, "acquire", agent=src1, patients=(tv1,))

    g2 = Graph(id="g2")
    src2 = g2.add_entity("Axiom Corp")
    tv2 = g2.add_entity("CloudScale")
    fact(g2, "acquire", agent=src2, patients=(tv2,))

    # Premise: the observed pair has no name signal
    idf = build_idf(["Axiom Corp", "VaultWorks", "CloudScale"])
    assert soft_tfidf("VaultWorks", "CloudScale", idf) < 0.1

    confidence, _, _ = match_graphs([g1, g2])

    assert confidence[(tv1.id, tv2.id)] > 0, (
        "patient-side path did not propagate anchor confidence"
    )


# ---------------------------------------------------------------------------
# Functionality weighting
# ---------------------------------------------------------------------------


def test_hub_participant_weakens_evidence():
    """A participant name that appears in many events is a weak identity
    signal: pooling it with hub occurrences lowers inverse functionality
    and with it the confidence of the event pair it anchors."""
    # Run A: clean two-graph acquisition, 2 shared participants
    a1 = Graph(id="a1")
    acme_a1 = a1.add_entity("Acme Corp")
    gamma_a1 = a1.add_entity("Gamma AI")
    acq_a = fact(a1, "acquire", agent=acme_a1, patients=(gamma_a1,))
    a2 = Graph(id="a2")
    acme_a2 = a2.add_entity("Acme Corp")
    gamma_a2 = a2.add_entity("Gamma AI")
    acq_b = fact(a2, "purchase", agent=acme_a2, patients=(gamma_a2,))

    conf_a, _, _ = match_graphs([a1, a2])

    # Run B: same pair plus background graphs in which "Gamma AI" is the
    # patient of many unrelated events — a hub.
    b1 = Graph(id="b1")
    acme_b1 = b1.add_entity("Acme Corp")
    gamma_b1 = b1.add_entity("Gamma AI")
    acq_c = fact(b1, "acquire", agent=acme_b1, patients=(gamma_b1,))
    b2 = Graph(id="b2")
    acme_b2 = b2.add_entity("Acme Corp")
    gamma_b2 = b2.add_entity("Gamma AI")
    acq_d = fact(b2, "purchase", agent=acme_b2, patients=(gamma_b2,))

    background = []
    for i in range(6):
        bg = Graph(id=f"bg{i}")
        hub = bg.add_entity("Gamma AI")
        other = bg.add_entity(f"Investor {i}")
        fact(bg, "invest in", agent=other, patients=(hub,))
        background.append(bg)

    conf_b, _, _ = match_graphs([b1, b2, *background])

    assert conf_b[(acq_c.id, acq_d.id)] < conf_a[(acq_a.id, acq_b.id)], (
        f"hub patient should weaken event evidence: "
        f"clean={conf_a[(acq_a.id, acq_b.id)]:.3f}, hub={conf_b[(acq_c.id, acq_d.id)]:.3f}"
    )


# ---------------------------------------------------------------------------
# Multi-hop propagation
# ---------------------------------------------------------------------------


def _two_hop_chain_graphs(
    anchors: list[str],
) -> tuple[Graph, Graph, object, object, object, object]:
    """Build a two-graph chain with dissimilar far and mid names.

    far --patient-- event --agent-- mid, and mid is the agent of one event
    per identical-name anchor. Returns (g1, g2, far1, mid1, far2, mid2)."""
    g1 = Graph(id="g1")
    far1 = g1.add_entity("Cordovan Industries")
    mid1 = g1.add_entity("Alpha Corp")
    fact(g1, "acquire", agent=mid1, patients=(far1,))
    for name in anchors:
        fact(g1, "partner with", agent=mid1, patients=(g1.add_entity(name),))

    g2 = Graph(id="g2")
    far2 = g2.add_entity("NexGen Holdings")
    mid2 = g2.add_entity("Beta Inc")
    fact(g2, "purchase", agent=mid2, patients=(far2,))
    for name in anchors:
        fact(g2, "collaborate with", agent=mid2, patients=(g2.add_entity(name),))

    return g1, g2, far1, mid1, far2, mid2


_THREE_ANCHORS = ["James Chen", "Austin", "DataVault"]


def test_multi_hop_propagation_across_iterations():
    """Evidence propagates along a chain of dissimilar names.

    Cordovan Industries / NexGen Holdings (far) can only be reached
    through Alpha Corp / Beta Inc (mid), which match only through three
    events with identical-name anchor patients. Both observed pairs have
    zero name similarity, so all confidence must arrive structurally.

    The mid pair needs three matched events because one matched event is
    a single path — and single paths never cross the merge bar."""
    g1, g2, far1, mid1, far2, mid2 = _two_hop_chain_graphs(_THREE_ANCHORS)

    # Premise: mid and far pairs have no name signal
    idf = build_idf(
        [
            "Cordovan Industries",
            "Alpha Corp",
            "James Chen",
            "Austin",
            "DataVault",
            "NexGen Holdings",
            "Beta Inc",
        ]
    )
    assert soft_tfidf("Alpha Corp", "Beta Inc", idf) < 0.1
    assert soft_tfidf("Cordovan Industries", "NexGen Holdings", idf) < 0.1

    confidence, _, _ = match_graphs([g1, g2])

    assert confidence[(mid1.id, mid2.id)] > 0, "mid pair not boosted by anchors"
    assert confidence[(far1.id, far2.id)] > 0, "2-hop propagation failed for far pair"
    # Evidence attenuates with distance from the anchors
    assert confidence[(mid1.id, mid2.id)] > confidence[(far1.id, far2.id)]


# ---------------------------------------------------------------------------
# Name variation with structural reinforcement
# ---------------------------------------------------------------------------


def test_name_variation_with_structural_reinforcement():
    """The core use case: similar-but-not-identical entity names get
    matched when structural evidence reinforces them."""
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault Inc")
    ceo1 = g1.add_entity("Elena Vasquez")
    fact(g1, "acquire", agent=meridian1, patients=(dv1,))
    fact(g1, "employ", agent=meridian1, patients=(ceo1,))

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault Inc")
    ceo2 = g2.add_entity("Elena Vasquez")
    fact(g2, "purchase", agent=meridian2, patients=(dv2,))
    fact(g2, "employ", agent=meridian2, patients=(ceo2,))

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    matched_pairs = set(matches)
    assert (meridian1.id, meridian2.id) in matched_pairs or (
        meridian2.id,
        meridian1.id,
    ) in matched_pairs
    assert (dv1.id, dv2.id) in matched_pairs or (dv2.id, dv1.id) in matched_pairs


# ---------------------------------------------------------------------------
# Dangling entities
# ---------------------------------------------------------------------------


def test_dangling_entities_get_no_boost():
    """Entities whose only shared event does not match should not be
    boosted, even when other entities match."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    solar = g1.add_entity("SolarGrid")
    fact(g1, "acquire", agent=apple1, patients=(beats1,))
    fact(g1, "hire", agent=apple1, patients=(solar,))

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    wind = g2.add_entity("WindPower")
    fact(g2, "purchase", agent=apple2, patients=(beats2,))
    fact(g2, "hire", agent=apple2, patients=(wind,))

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    matched_ids = {entity_id for pair in matches for entity_id in pair}
    assert solar.id not in matched_ids
    assert wind.id not in matched_ids


# ---------------------------------------------------------------------------
# Multiple matched events accumulate
# ---------------------------------------------------------------------------


def test_multiple_matched_events_accumulate():
    """An entity pair participating in two matched events accumulates more
    evidence than one participating in a single matched event."""
    # Single shared event
    g1u = Graph(id="g1u")
    m1u = g1u.add_entity("Meridian Technologies")
    dv1u = g1u.add_entity("DataVault")
    fact(g1u, "acquire", agent=m1u, patients=(dv1u,))

    g2u = Graph(id="g2u")
    m2u = g2u.add_entity("Meridian Tech")
    dv2u = g2u.add_entity("DataVault")
    fact(g2u, "purchase", agent=m2u, patients=(dv2u,))

    confidence_uni, _, _ = match_graphs([g1u, g2u])

    # Two shared events
    g1b = Graph(id="g1b")
    m1b = g1b.add_entity("Meridian Technologies")
    dv1b = g1b.add_entity("DataVault")
    ceo1b = g1b.add_entity("Elena Vasquez")
    fact(g1b, "acquire", agent=m1b, patients=(dv1b,))
    fact(g1b, "employ", agent=m1b, patients=(ceo1b,))

    g2b = Graph(id="g2b")
    m2b = g2b.add_entity("Meridian Tech")
    dv2b = g2b.add_entity("DataVault")
    ceo2b = g2b.add_entity("Elena Vasquez")
    fact(g2b, "purchase", agent=m2b, patients=(dv2b,))
    fact(g2b, "employ", agent=m2b, patients=(ceo2b,))

    confidence_bi, _, _ = match_graphs([g1b, g2b])

    assert confidence_bi[(m1b.id, m2b.id)] > confidence_uni[(m1u.id, m2u.id)], (
        f"two matched events ({confidence_bi[(m1b.id, m2b.id)]:.3f}) should beat "
        f"one ({confidence_uni[(m1u.id, m2u.id)]:.3f})"
    )


# ---------------------------------------------------------------------------
# Structural override of name dissimilarity
# ---------------------------------------------------------------------------


def test_shared_event_does_not_override_name_dissimilarity():
    """A shared high-confidence patient should not cause agents with
    different names to match.

    NovaTech Labs has identical names across graphs, but Dr. Priya Sharma
    and Dr. Elena Vasquez are different people: a single event path cannot
    overcome name dissimilarity."""
    g1 = Graph(id="g1")
    sharma = g1.add_entity("Dr. Priya Sharma")
    nova1 = g1.add_entity("NovaTech Labs")
    fact(g1, "found", agent=sharma, patients=(nova1,))

    g2 = Graph(id="g2")
    vasquez = g2.add_entity("Dr. Elena Vasquez")
    nova2 = g2.add_entity("NovaTech Labs")
    fact(g2, "found", agent=vasquez, patients=(nova2,))

    # Premise: name similarity alone is below threshold
    idf = build_idf(["Dr. Priya Sharma", "Dr. Elena Vasquez", "NovaTech Labs"])
    assert soft_tfidf("Dr. Priya Sharma", "Dr. Elena Vasquez", idf) < 0.8

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    matched_pairs = set(matches)
    assert (sharma.id, vasquez.id) not in matched_pairs and (
        vasquez.id,
        sharma.id,
    ) not in matched_pairs, (
        "Structural evidence from shared patient overrode name dissimilarity"
    )


def test_similar_names_disjoint_neighborhoods_no_match():
    """Near-identical names with zero structural overlap should not match.

    Replicates the Elena/Lena Vasquez false merge from real data. Their
    CEO events share only the (name-mismatched) agent path, and the
    mismatched company patients suppress the event pair."""
    g1 = Graph(id="g1")
    elena = g1.add_entity("Dr. Elena Vasquez")
    volta = g1.add_entity("Volta Systems")
    fact(g1, "be CEO of", agent=elena, patients=(volta,))

    g2 = Graph(id="g2")
    lena = g2.add_entity("Dr. Lena Vasquez")
    halcyon = g2.add_entity("Halcyon Genomics")
    fact(g2, "be CEO of", agent=lena, patients=(halcyon,))

    # Premise: neighbor names have no similarity
    idf = build_idf(["Volta Systems", "Halcyon Genomics"])
    assert soft_tfidf("Volta Systems", "Halcyon Genomics", idf) < 0.5

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    matched_pairs = set(matches)

    assert (elena.id, lena.id) not in matched_pairs and (
        lena.id,
        elena.id,
    ) not in matched_pairs, (
        "Similar names with disjoint neighborhoods were incorrectly matched"
    )


# ---------------------------------------------------------------------------
# Convergence / early stopping
# ---------------------------------------------------------------------------


def test_simple_graph_stabilizes_well_before_max_iter():
    """A simple two-entity graph reaches its fixed point well before
    max_iter=30: max_iter=25 and max_iter=30 agree to within 1e-6."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    fact(g1, "acquire", agent=apple1, patients=(beats1,))

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    fact(g2, "acquire", agent=apple2, patients=(beats2,))

    conf_25, _, _ = match_graphs([g1, g2], max_iter=25)
    conf_30, _, _ = match_graphs([g1, g2], max_iter=30)

    assert conf_25[(apple1.id, apple2.id)] == pytest.approx(
        conf_30[(apple1.id, apple2.id)], abs=1e-6
    )
    assert conf_25[(beats1.id, beats2.id)] == pytest.approx(
        conf_30[(beats1.id, beats2.id)], abs=1e-6
    )


def test_multi_hop_needs_multiple_iterations():
    """One iteration cannot reach the far end of a chain.

    Iteration 1 lifts only the anchor-adjacent events; the mid pair moves
    only once those events cross 0.5, and the far pair one step later.
    max_iter=1 must leave the far pair at exactly its seed (0.0)."""
    g1, g2, far1, mid1, far2, mid2 = _two_hop_chain_graphs(_THREE_ANCHORS)

    conf_1, _, _ = match_graphs([g1, g2], max_iter=1)
    conf_30, _, _ = match_graphs([g1, g2], max_iter=30)

    assert conf_1[(far1.id, far2.id)] == 0.0, (
        "far pair boosted in a single iteration — test scenario is not multi-hop"
    )
    assert conf_30[(far1.id, far2.id)] > conf_1[(far1.id, far2.id)]
    assert conf_30[(mid1.id, mid2.id)] > conf_1[(mid1.id, mid2.id)]


def test_propagation_converges():
    """Running with more iterations than needed does not change results."""
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault Inc")
    ceo1 = g1.add_entity("Elena Vasquez")
    fact(g1, "acquire", agent=meridian1, patients=(dv1,))
    fact(g1, "employ", agent=meridian1, patients=(ceo1,))

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault Inc")
    ceo2 = g2.add_entity("Elena Vasquez")
    fact(g2, "purchase", agent=meridian2, patients=(dv2,))
    fact(g2, "employ", agent=meridian2, patients=(ceo2,))

    conf_25, _, _ = match_graphs([g1, g2], max_iter=25)
    conf_30, _, _ = match_graphs([g1, g2], max_iter=30)
    for pair, val in conf_30.items():
        assert conf_25.get(pair, 0.0) == pytest.approx(val, abs=1e-6), (
            f"Score changed between max_iter=25 and max_iter=30 for {pair}: "
            f"{conf_25.get(pair, 0.0):.6f} → {val:.6f}"
        )


# ---------------------------------------------------------------------------
# Pair space
# ---------------------------------------------------------------------------


def test_same_graph_entities_never_match():
    """Two entities within the same article graph never appear as a pair,
    regardless of name similarity."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple Inc")
    music = g.add_entity("Apple Music")
    fact(g, "own", agent=apple, patients=(music,))

    g2 = Graph(id="g2")
    g2.add_entity("Google")
    g2.add_entity("Alphabet")

    confidence, _, _ = match_graphs([g, g2])

    assert (apple.id, music.id) not in confidence
    assert (music.id, apple.id) not in confidence


def test_single_graph_produces_no_matches():
    """match_graphs with a single graph returns an empty confidence dict."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    fact(g, "acquire", agent=apple, patients=(beats,))

    confidence, _, _ = match_graphs([g])

    assert confidence == {}


def test_entities_and_events_never_pair():
    """Pairs are formed only within a kind: no entity-event confidence
    entries ever exist, however suggestive the names."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    event1 = fact(g1, "acquire", agent=apple1)

    g2 = Graph(id="g2")
    # An entity named exactly like g1's event label — still never paired
    # with the event, only with same-kind nodes.
    g2.add_entity("acquire")
    g2.add_entity("Apple")

    confidence, _, _ = match_graphs([g1, g2])

    for a, b in confidence:
        assert a != event1.id and b != event1.id, (
            f"event node appeared in a confidence pair: {(a, b)}"
        )


# ---------------------------------------------------------------------------
# Multi-label name seeding
# ---------------------------------------------------------------------------


def test_multi_label_entity_uses_best_name_pair():
    """An entity with multiple names seeds similarity from the best name
    pair across both entities' name lists.

    "Meridian Technologies" in g1, names=["Meridian Tech",
    "Meridian Technologies"] in g2. The best pair is the exact match
    (~1.0), not "Meridian Technologies"/"Meridian Tech" (~0.88)."""
    g1 = Graph(id="g1")
    m1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault")
    fact(g1, "acquire", agent=m1, patients=(dv1,))

    g2 = Graph(id="g2")
    m2 = g2.add_entity(["Meridian Tech", "Meridian Technologies"])
    dv2 = g2.add_entity("DataVault")
    fact(g2, "purchase", agent=m2, patients=(dv2,))

    confidence, _, _ = match_graphs([g1, g2])

    assert confidence[(m1.id, m2.id)] > 0.8


def test_multi_label_all_names_contribute_to_idf():
    """All names in an entity's name list contribute to IDF computation."""
    g1 = Graph(id="g1")
    m1 = g1.add_entity(["Meridian Technologies", "Meridian Tech"])
    dv1 = g1.add_entity("DataVault")
    fact(g1, "acquire", agent=m1, patients=(dv1,))

    g2 = Graph(id="g2")
    m2 = g2.add_entity("Meridian Technologies")
    dv2 = g2.add_entity("DataVault")
    fact(g2, "purchase", agent=m2, patients=(dv2,))

    confidence, _, _ = match_graphs([g1, g2])
    assert confidence[(m1.id, m2.id)] > 0.8


# ---------------------------------------------------------------------------
# Progressive merging — enriched neighborhood
# ---------------------------------------------------------------------------


def test_progressive_merging_enriched_neighborhood():
    """Progressive merging enriches neighborhoods across merge rounds,
    enabling matches that pairwise comparison alone cannot produce.

    Articles A and B describe Meridian Corp with overlapping events
    (acquire DataVault, employ James Chen, alumna Stanford) plus unique
    ones (A: headquartered in Austin; B: partnered with Volta Systems).
    They merge in the first round.

    Article C describes "Meridian Tech Corp" — moderate name similarity
    to "Meridian Corp" — with events sharing Austin (A-only), Volta
    (B-only), and James Chen (both). Only after A+B merge does C see all
    three matched events against a single counterpart, producing strictly
    higher confidence than any single pairwise comparison.
    """
    ga = Graph(id="a")
    ma = ga.add_entity("Meridian Corp")
    dva = ga.add_entity("DataVault")
    ja = ga.add_entity("James Chen")
    su_a = ga.add_entity("Stanford University")
    austin_a = ga.add_entity("Austin")
    fact(ga, "acquire", agent=ma, patients=(dva,))
    fact(ga, "employ", agent=ma, patients=(ja,))
    fact(ga, "be alumna of", agent=ma, patients=(su_a,))
    fact(ga, "be headquartered in", agent=ma, patients=(austin_a,))

    gb = Graph(id="b")
    mb = gb.add_entity("Meridian Corp")
    dvb = gb.add_entity("DataVault")
    jb = gb.add_entity("James Chen")
    su_b = gb.add_entity("Stanford University")
    volta_b = gb.add_entity("Volta Systems")
    fact(gb, "purchase", agent=mb, patients=(dvb,))
    fact(gb, "employ", agent=mb, patients=(jb,))
    fact(gb, "be alumna of", agent=mb, patients=(su_b,))
    fact(gb, "partner with", agent=mb, patients=(volta_b,))

    gc = Graph(id="c")
    mc = gc.add_entity("Meridian Tech Corp")
    austin_c = gc.add_entity("Austin")
    volta_c = gc.add_entity("Volta Systems")
    jc = gc.add_entity("James Chen")
    fact(gc, "be headquartered in", agent=mc, patients=(austin_c,))
    fact(gc, "partner with", agent=mc, patients=(volta_c,))
    fact(gc, "employ", agent=mc, patients=(jc,))

    graphs = [ga, gb, gc]

    # Premise: name similarity alone is insufficient
    names = [name for g in graphs for n in g.nodes.values() for name in n.names]
    idf = build_idf(names)
    assert soft_tfidf("Meridian Tech Corp", "Meridian Corp", idf) < 0.8

    # Premise: A+B merge above the merge threshold
    conf_single, _, _ = match_graphs(graphs, merge_threshold=float("inf"))
    assert conf_single[(ma.id, mb.id)] >= 0.7, (
        f"A-B should merge: {conf_single[(ma.id, mb.id)]:.3f}"
    )

    # Without progressive merging (merge_threshold=inf), C sees only pairwise evidence
    conf_progressive, _, _ = match_graphs(graphs)

    c_single = max(
        conf_single.get((mc.id, ma.id), 0),
        conf_single.get((mc.id, mb.id), 0),
    )
    c_progressive = max(
        conf_progressive.get((mc.id, ma.id), 0),
        conf_progressive.get((mc.id, mb.id), 0),
    )
    assert c_progressive > c_single, (
        f"Progressive merging should improve C's match: "
        f"single={c_single:.3f}, progressive={c_progressive:.3f}"
    )


# ---------------------------------------------------------------------------
# Negative evidence vs structural matches
# ---------------------------------------------------------------------------


def test_negative_evidence_does_not_over_penalize_structurally_matched_neighbors():
    """Negative evidence should not penalize an entity pair when its
    neighbors are structurally matched despite weak name similarity.

    "Meridian Technologies" and "Meridian Tech" share two matched events:
    acquire/purchase of DataVault, and a CEO event whose person names are
    weak ('Dr. Alice M. Johnson' / 'A. Johnson') but who share a second
    event (graduated from Stanford University).

    Propagation discovers the CEO match via Stanford. Even though the CEO
    names are dissimilar, the structural evidence from the shared Stanford
    event outweighs the negative signal from the name mismatch."""
    g1 = Graph(id="g1")
    m1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault")
    ceo1 = g1.add_entity("Dr. Alice M. Johnson")
    uni1 = g1.add_entity("Stanford University")
    fact(g1, "acquire", agent=m1, patients=(dv1,))
    fact(g1, "employ as CEO", agent=m1, patients=(ceo1,))
    fact(g1, "graduate from", agent=ceo1, patients=(uni1,))

    g2 = Graph(id="g2")
    m2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault")
    ceo2 = g2.add_entity("A. Johnson")
    uni2 = g2.add_entity("Stanford University")
    fact(g2, "purchase", agent=m2, patients=(dv2,))
    fact(g2, "employ as CEO", agent=m2, patients=(ceo2,))
    fact(g2, "graduate from", agent=ceo2, patients=(uni2,))

    graphs = [g1, g2]

    # Premise: CEO name similarity is weak (structural propagation needed)
    all_names = [n for g in graphs for node in g.nodes.values() for n in node.names]
    idf = build_idf(all_names)
    assert soft_tfidf("Dr. Alice M. Johnson", "A. Johnson", idf) < 0.5

    confidence, _, _ = match_graphs(graphs)

    # CEO pair should be structurally matched despite weak names
    ceo_score = confidence.get(
        (ceo1.id, ceo2.id), confidence.get((ceo2.id, ceo1.id), 0.0)
    )
    assert ceo_score > 0.6, (
        f"CEO pair should be structurally matched, got {ceo_score:.3f}"
    )

    # Meridian should not be over-penalized — the CEO pair matches
    # structurally via Stanford.
    meridian_score = confidence.get((m1.id, m2.id), confidence.get((m2.id, m1.id), 0.0))
    assert meridian_score > 0.8, (
        f"Negative evidence over-penalized Meridian: score={meridian_score:.3f} "
        f"(CEO structural match={ceo_score:.3f})"
    )


# ---------------------------------------------------------------------------
# Synonym relation inflation (false merges from real data)
# ---------------------------------------------------------------------------


def test_predecessor_successor_at_same_company_no_match():
    """A CEO transition at the same company reported by two sources with
    different phrasings should merge same-name entities while keeping the
    predecessor and successor separate.

    Park's CEO event and Chen's named-CEO event share the company patient
    but have name-mismatched agents — the cross events suppress, so
    Park↔Chen stays below the merge threshold while the same-name pairs
    merge. Reproduces the David Park / Sarah Chen pattern from real data."""
    g1 = Graph(id="g1")
    park1 = g1.add_entity("David Park")
    chen1 = g1.add_entity("Sarah Chen")
    nextera1 = g1.add_entity("Nextera Energy Solutions")
    fact(g1, "be CEO of", agent=park1, patients=(nextera1,))
    fact(g1, "be named CEO of", agent=chen1, patients=(nextera1,))

    g2 = Graph(id="g2")
    park2 = g2.add_entity("David Park")
    chen2 = g2.add_entity("Sarah Chen")
    nextera2 = g2.add_entity("Nextera Energy Solutions")
    fact(g2, "serve as CEO of", agent=park2, patients=(nextera2,))
    fact(g2, "become CEO of", agent=chen2, patients=(nextera2,))

    confidence, _, _ = match_graphs([g1, g2])

    matches = _select_matches(confidence, threshold=0.8)
    park_ids = {park1.id, park2.id}
    chen_ids = {chen1.id, chen2.id}
    assert any(id_a in park_ids and id_b in park_ids for id_a, id_b in matches), (
        "Same-name Park entities should match"
    )
    assert any(id_a in chen_ids and id_b in chen_ids for id_a, id_b in matches), (
        "Same-name Chen entities should match"
    )

    for id_a, id_b in matches:
        assert not (
            (id_a in park_ids and id_b in chen_ids)
            or (id_a in chen_ids and id_b in park_ids)
        ), "Predecessor and successor CEOs incorrectly matched"


def test_shared_summit_does_not_merge_different_people():
    """Two different people who both spoke at the same conference should
    not be merged, even with a weak name prefix match ('Dr.').

    Dr. Vasquez is CTO of Volta (semiconductor company).
    Dr. Sharma founded Lightwave (analytics startup).
    Multiple sources report each person at TechForward Summit with
    different event labels ('speak at' / 'attend' / 'give keynote at').

    Reproduces the Vasquez / Sharma false merge from real data."""
    g1 = Graph(id="g1")
    vasquez1 = g1.add_entity("Dr. Elena Vasquez")
    volta1 = g1.add_entity("Volta Systems")
    summit1 = g1.add_entity("TechForward Summit")
    fact(g1, "be installed as CTO of", agent=vasquez1, patients=(volta1,))
    fact(g1, "speak at", agent=vasquez1, patients=(summit1,))

    g2 = Graph(id="g2")
    sharma2 = g2.add_entity("Dr. Priya Sharma")
    lightwave2 = g2.add_entity("Lightwave Analytics")
    summit2 = g2.add_entity("TechForward")
    fact(g2, "be founder of", agent=sharma2, patients=(lightwave2,))
    fact(g2, "attend", agent=sharma2, patients=(summit2,))

    g3 = Graph(id="g3")
    vasquez3 = g3.add_entity("Dr. Elena Vasquez")
    volta3 = g3.add_entity("Volta Systems")
    ibm3 = g3.add_entity("IBM Research")
    summit3 = g3.add_entity("TechForward Summit")
    fact(g3, "be CTO of", agent=vasquez3, patients=(volta3,))
    fact(g3, "work at", agent=vasquez3, patients=(ibm3,))
    fact(g3, "give keynote at", agent=vasquez3, patients=(summit3,))

    g4 = Graph(id="g4")
    sharma4 = g4.add_entity("Dr. Priya Sharma")
    lightwave4 = g4.add_entity("Lightwave Analytics")
    meridian4 = g4.add_entity("Meridian Technologies")
    summit4 = g4.add_entity("TechForward Summit")
    fact(g4, "co-found", agent=sharma4, patients=(lightwave4,))
    fact(g4, "become SVP of Analytics at", agent=sharma4, patients=(meridian4,))
    fact(g4, "give keynote at", agent=sharma4, patients=(summit4,))

    confidence, _, _ = match_graphs([g1, g2, g3, g4])

    matches = _select_matches(confidence, threshold=0.8)
    vasquez_ids = {vasquez1.id, vasquez3.id}
    sharma_ids = {sharma2.id, sharma4.id}
    for id_a, id_b in matches:
        assert not (
            (id_a in vasquez_ids and id_b in sharma_ids)
            or (id_a in sharma_ids and id_b in vasquez_ids)
        ), "Different people who spoke at same event incorrectly matched"
