"""Layer 2 tests for similarity propagation.

These tests exercise match_graphs() on small graphs with real embeddings.
Tests verify that:

- Matching names + matching relations produce correct matches
- Synonym relations propagate through the relation gate
- Dissimilar relations, weak neighbors, and many weak paths produce no
  spurious matches (GH issue #1)
- Incoming edges propagate evidence (not just outgoing)
- Functionality weighting affects evidence strength
- Multi-hop chains require iterative propagation
- Name variation with structural reinforcement (the core use case)
- Dangling entities get no structural evidence
- Exponential sum accumulates evidence from multiple paths (bidirectional > unidirectional)
"""

from worldgraph.graph import Graph
from worldgraph.match import match_graphs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_matches(confidence, threshold=0.8):
    """Select entity matches: pairs where confidence >= threshold."""
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


def test_matching_names_and_relations_produce_matches(embedder):
    """Two graphs with the same entity names and identical relations should
    produce high-confidence matches for the correct entity pairs."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    g1.add_edge(apple1, beats1, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    g2.add_edge(apple2, beats2, "acquired")

    confidence = match_graphs([g1, g2], embedder)

    assert confidence[(apple1.id, apple2.id)] > 0.8
    assert confidence[(beats1.id, beats2.id)] > 0.8

    assert confidence[(apple1.id, beats2.id)] < 0.5
    assert confidence[(beats1.id, apple2.id)] < 0.5


def test_synonym_relations_propagate(embedder):
    """Synonym relation phrases ('acquired' / 'purchased') should pass the
    relation gate and produce correct matches."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    g1.add_edge(apple1, beats1, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    g2.add_edge(apple2, beats2, "purchased")

    confidence = match_graphs([g1, g2], embedder)

    assert confidence[(apple1.id, apple2.id)] > 0.8
    assert confidence[(beats1.id, beats2.id)] > 0.8


# ---------------------------------------------------------------------------
# No spurious matches (GH issue #1)
# ---------------------------------------------------------------------------


def test_dissimilar_relations_do_not_propagate(embedder):
    """Unrelated relation phrases ('acquired' vs 'located in') should not
    pass the relation gate — no matches between unrelated graphs."""
    g1 = Graph(id="g1")
    apple = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    g1.add_edge(apple, beats, "acquired")

    g2 = Graph(id="g2")
    tokyo = g2.add_entity("Tokyo")
    japan = g2.add_entity("Japan")
    g2.add_edge(tokyo, japan, "located in")

    confidence = match_graphs([g1, g2], embedder)

    matches = _select_matches(confidence, threshold=0.8)
    assert matches == [], f"Spurious matches found: {matches}"


def test_weak_neighbors_do_not_produce_matches(embedder):
    """Even with identical relation phrases, weak neighbor similarity
    should not produce matches."""
    g1 = Graph(id="g1")
    apple = g1.add_entity("Apple")
    beats = g1.add_entity("Beats")
    g1.add_edge(apple, beats, "acquired")

    g2 = Graph(id="g2")
    google = g2.add_entity("Google")
    youtube = g2.add_entity("YouTube")
    g2.add_edge(google, youtube, "acquired")

    confidence = match_graphs([g1, g2], embedder)

    matches = _select_matches(confidence, threshold=0.8)
    assert matches == [], f"Spurious matches from weak neighbors: {matches}"


def test_many_weak_paths_do_not_accumulate(embedder):
    """Many unrelated edges should not produce matches."""
    g1 = Graph(id="g1")
    org = g1.add_entity("Org")
    target = g1.add_entity("Target")
    project = g1.add_entity("Project")
    person = g1.add_entity("Person")
    g1.add_edge(org, target, "acquired")
    g1.add_edge(org, project, "funded")
    g1.add_edge(org, person, "hired")

    g2 = Graph(id="g2")
    city = g2.add_entity("City")
    country = g2.add_entity("Country")
    river = g2.add_entity("River")
    event = g2.add_entity("Event")
    g2.add_edge(city, country, "located in")
    g2.add_edge(city, river, "borders")
    g2.add_edge(city, event, "hosts")

    confidence = match_graphs([g1, g2], embedder)

    matches = _select_matches(confidence, threshold=0.8)
    assert matches == [], f"Spurious matches from accumulated weak paths: {matches}"


# ---------------------------------------------------------------------------
# Incoming edges
# ---------------------------------------------------------------------------


def test_incoming_edges_propagate(embedder):
    """Structural evidence should propagate through incoming edges, not just
    outgoing.

    DataVault has identical names (high name sim). The incoming edge should
    propagate that confidence to the Meridian pair via exponential sum."""
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault")
    g1.add_edge(meridian1, dv1, "acquired")

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault")
    g2.add_edge(meridian2, dv2, "purchased")

    confidence = match_graphs([g1, g2], embedder)

    assert confidence[(meridian1.id, meridian2.id)] > 0, (
        "Incoming edge path did not propagate evidence to Meridian pair"
    )


# ---------------------------------------------------------------------------
# Functionality weighting
# ---------------------------------------------------------------------------


def test_functional_relation_produces_stronger_evidence(embedder):
    """A 1:1 (functional) relation should produce stronger confidence boost
    than a many-to-one relation, all else being equal.

    Background graphs establish that 'acquired' is 1:1 while 'invested in'
    has fan-in. Including them in match_graphs lets functionality stats
    reflect this without needing low-level API access."""
    # Background: 'acquired' is 1:1
    bg1 = Graph(id="bg1")
    bg1_apple = bg1.add_entity("Apple")
    bg1_beats = bg1.add_entity("Beats")
    bg1_google = bg1.add_entity("Google")
    bg1_yt = bg1.add_entity("YouTube")
    bg1.add_edge(bg1_apple, bg1_beats, "acquired")
    bg1.add_edge(bg1_google, bg1_yt, "acquired")

    # Background: 'invested in' has fan-in
    bg2 = Graph(id="bg2")
    bg2_apple = bg2.add_entity("Apple")
    bg2_google = bg2.add_entity("Google")
    bg2_ms = bg2.add_entity("Microsoft")
    bg2_dv = bg2.add_entity("DataVault")
    bg2.add_edge(bg2_apple, bg2_dv, "invested in")
    bg2.add_edge(bg2_google, bg2_dv, "invested in")
    bg2.add_edge(bg2_ms, bg2_dv, "invested in")

    # Test pair with 'acquired' (high inverse functionality)
    g1a = Graph(id="g1a")
    m1a = g1a.add_entity("Meridian Technologies")
    dv1a = g1a.add_entity("DataVault")
    g1a.add_edge(m1a, dv1a, "acquired")

    g2a = Graph(id="g2a")
    m2a = g2a.add_entity("Meridian Tech")
    dv2a = g2a.add_entity("DataVault")
    g2a.add_edge(m2a, dv2a, "acquired")

    confidence_acq = match_graphs([bg1, bg2, g1a, g2a], embedder)

    # Test pair with 'invested in' (low inverse functionality)
    g1i = Graph(id="g1i")
    m1i = g1i.add_entity("Meridian Technologies")
    dv1i = g1i.add_entity("DataVault")
    g1i.add_edge(m1i, dv1i, "invested in")

    g2i = Graph(id="g2i")
    m2i = g2i.add_entity("Meridian Tech")
    dv2i = g2i.add_entity("DataVault")
    g2i.add_edge(m2i, dv2i, "invested in")

    confidence_inv = match_graphs([bg1, bg2, g1i, g2i], embedder)

    assert confidence_acq[(m1a.id, m2a.id)] > confidence_inv[(m1i.id, m2i.id)], (
        f"Functional relation confidence ({confidence_acq[(m1a.id, m2a.id)]}) should exceed "
        f"non-functional ({confidence_inv[(m1i.id, m2i.id)]})"
    )


# ---------------------------------------------------------------------------
# Multi-hop propagation
# ---------------------------------------------------------------------------


def test_multi_hop_propagation_across_iterations(embedder):
    """Evidence propagates through a chain: a high-confidence anchor at
    the end boosts intermediate nodes, which in turn boost further nodes.

    James Chen pair has identical names → high name sim.
    Alpha Corp / Beta Inc have very low name similarity — they can only be
    matched through their shared, confidently-matched neighbor (James Chen).
    """
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    alpha = g1.add_entity("Alpha Corp")
    james1 = g1.add_entity("James Chen")
    g1.add_edge(meridian1, alpha, "acquired")
    g1.add_edge(alpha, james1, "founded by")

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    beta = g2.add_entity("Beta Inc")
    james2 = g2.add_entity("James Chen")
    g2.add_edge(meridian2, beta, "purchased")
    g2.add_edge(beta, james2, "founded by")

    confidence = match_graphs([g1, g2], embedder)

    # Alpha Corp / Beta Inc boosted by James Chen chain
    assert confidence[(alpha.id, beta.id)] > 0, (
        "Alpha Corp / Beta Inc not boosted despite shared James Chen neighbor"
    )

    # Meridian pair boosted via the full chain
    assert confidence[(meridian1.id, meridian2.id)] > 0, (
        "Multi-hop propagation failed: Meridian pair not boosted after convergence"
    )


# ---------------------------------------------------------------------------
# Name variation with structural reinforcement
# ---------------------------------------------------------------------------


def test_name_variation_with_structural_reinforcement(embedder):
    """The core use case: similar-but-not-identical entity names get matched
    when structural evidence reinforces them."""
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault Inc")
    ceo1 = g1.add_entity("Elena Vasquez")
    g1.add_edge(meridian1, dv1, "acquired")
    g1.add_edge(meridian1, ceo1, "hired")

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault Inc")
    ceo2 = g2.add_entity("Elena Vasquez")
    g2.add_edge(meridian2, dv2, "purchased")
    g2.add_edge(meridian2, ceo2, "employed")

    confidence = match_graphs([g1, g2], embedder)

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


def test_dangling_entities_get_no_boost(embedder):
    """Entities with no matching structure should not be boosted beyond
    their name-similarity seed, even when other entities match."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    solar = g1.add_entity("SolarGrid")
    g1.add_edge(apple1, beats1, "acquired")
    g1.add_edge(apple1, solar, "hired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    wind = g2.add_entity("WindPower")
    g2.add_edge(apple2, beats2, "purchased")
    g2.add_edge(apple2, wind, "hired")

    confidence = match_graphs([g1, g2], embedder)

    matches = _select_matches(confidence, threshold=0.8)
    matched_ids = {entity_id for pair in matches for entity_id in pair}
    assert solar.id not in matched_ids
    assert wind.id not in matched_ids


# ---------------------------------------------------------------------------
# Exp-sum accumulation (bidirectional > unidirectional)
# ---------------------------------------------------------------------------


def test_bidirectional_edges_accumulate(embedder):
    """Entity pairs connected by edges in both directions should accumulate
    evidence, producing higher confidence than a single edge."""
    # Unidirectional
    g1u = Graph(id="g1u")
    m1u = g1u.add_entity("Meridian Technologies")
    dv1u = g1u.add_entity("DataVault")
    g1u.add_edge(m1u, dv1u, "acquired")

    g2u = Graph(id="g2u")
    m2u = g2u.add_entity("Meridian Tech")
    dv2u = g2u.add_entity("DataVault")
    g2u.add_edge(m2u, dv2u, "acquired")

    confidence_uni = match_graphs([g1u, g2u], embedder)

    # Bidirectional
    g1b = Graph(id="g1b")
    m1b = g1b.add_entity("Meridian Technologies")
    dv1b = g1b.add_entity("DataVault")
    g1b.add_edge(m1b, dv1b, "acquired")
    g1b.add_edge(dv1b, m1b, "subsidiary of")

    g2b = Graph(id="g2b")
    m2b = g2b.add_entity("Meridian Tech")
    dv2b = g2b.add_entity("DataVault")
    g2b.add_edge(m2b, dv2b, "acquired")
    g2b.add_edge(dv2b, m2b, "subsidiary of")

    confidence_bi = match_graphs([g1b, g2b], embedder)

    assert confidence_bi[(m1b.id, m2b.id)] >= confidence_uni[(m1u.id, m2u.id)], (
        f"Bidirectional ({confidence_bi[(m1b.id, m2b.id)]}) should be >= "
        f"unidirectional ({confidence_uni[(m1u.id, m2u.id)]})"
    )


# ---------------------------------------------------------------------------
# Structural override of name dissimilarity
# ---------------------------------------------------------------------------


def test_shared_anchor_does_not_override_name_dissimilarity(embedder):
    """A shared high-confidence anchor should not cause entities with
    different names to match.

    NovaTech Labs has identical name sim across graphs. "founded"
    passes the relation gate. But under exponential sum, a single
    structural path is insufficient to cross the threshold."""
    g1 = Graph(id="g1")
    sharma = g1.add_entity("Dr. Priya Sharma")
    nova1 = g1.add_entity("NovaTech Labs")
    g1.add_edge(sharma, nova1, "founded")

    g2 = Graph(id="g2")
    vasquez = g2.add_entity("Dr. Elena Vasquez")
    nova2 = g2.add_entity("NovaTech Labs")
    g2.add_edge(vasquez, nova2, "founded")

    # Background: establish "founded" as a functional (1:1) relation
    bg_graphs = []
    for i, (person, org) in enumerate(
        [
            ("Marcus Webb", "Alpha Corp"),
            ("Sarah Chen", "Beta Inc"),
            ("James Xu", "Gamma LLC"),
        ]
    ):
        bg = Graph(id=f"bg{i}")
        p = bg.add_entity(person)
        o = bg.add_entity(org)
        bg.add_edge(p, o, "founded")
        bg_graphs.append(bg)

    # Premise: name similarity alone is below threshold
    from worldgraph.names import build_idf, soft_tfidf

    labels = [
        n.label
        for g in [g1, g2, *bg_graphs]
        for n in g.nodes.values()
        if hasattr(n, "label")
    ]
    idf = build_idf(labels)
    sv_name_sim = soft_tfidf("Dr. Priya Sharma", "Dr. Elena Vasquez", idf)
    assert sv_name_sim < 0.8

    confidence = match_graphs([g1, g2, *bg_graphs], embedder)

    matches = _select_matches(confidence, threshold=0.8)
    matched_pairs = set(matches)
    assert (nova1.id, nova2.id) not in matched_pairs and (
        nova2.id,
        nova1.id,
    ) not in matched_pairs, (
        "Identical names with no structural corroboration were incorrectly matched"
    )
    assert (sharma.id, vasquez.id) not in matched_pairs and (
        vasquez.id,
        sharma.id,
    ) not in matched_pairs, (
        "Structural evidence from shared anchor overrode name dissimilarity"
    )


def test_similar_names_disjoint_neighborhoods_no_match(embedder):
    """Near-identical names with zero structural overlap should not match.

    Replicates the Elena/Lena Vasquez false merge from real data."""
    g1 = Graph(id="g1")
    elena = g1.add_entity("Dr. Elena Vasquez")
    volta = g1.add_entity("Volta Systems")
    g1.add_edge(elena, volta, "is CEO of")

    g2 = Graph(id="g2")
    lena = g2.add_entity("Dr. Lena Vasquez")
    halcyon = g2.add_entity("Halcyon Genomics")
    g2.add_edge(lena, halcyon, "is CEO of")

    # Premise: neighbor names have no similarity
    from worldgraph.names import build_idf, soft_tfidf

    idf = build_idf(["Volta Systems", "Halcyon Genomics"])
    nbr_sim = soft_tfidf("Volta Systems", "Halcyon Genomics", idf)
    assert nbr_sim < 0.5

    confidence = match_graphs([g1, g2], embedder)

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


def test_simple_graph_stabilizes_well_before_max_iter(embedder):
    """A simple two-entity graph should reach a fixed point well before
    max_iter=30. Comparing max_iter=10 vs max_iter=30 should give identical
    results (mutual reinforcement between the two pairs converges quickly)."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    beats1 = g1.add_entity("Beats")
    g1.add_edge(apple1, beats1, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    beats2 = g2.add_entity("Beats")
    g2.add_edge(apple2, beats2, "acquired")

    conf_10 = match_graphs([g1, g2], embedder, max_iter=10)
    conf_30 = match_graphs([g1, g2], embedder, max_iter=30)

    assert conf_10[(apple1.id, apple2.id)] == conf_30[(apple1.id, apple2.id)]
    assert conf_10[(beats1.id, beats2.id)] == conf_30[(beats1.id, beats2.id)]


def test_multi_hop_needs_multiple_iterations(embedder):
    """A chain graph needs multiple iterations — max_iter=1 should produce
    lower confidence for the far end than max_iter=30."""
    g1 = Graph(id="g1")
    a1 = g1.add_entity("Alpha Corp")
    b1 = g1.add_entity("James Chen")
    c1 = g1.add_entity("DataVault")
    g1.add_edge(a1, b1, "founded by")
    g1.add_edge(b1, c1, "leads")

    g2 = Graph(id="g2")
    a2 = g2.add_entity("Alpha Corp")
    b2 = g2.add_entity("James Chen")
    c2 = g2.add_entity("DataVault")
    g2.add_edge(a2, b2, "founded by")
    g2.add_edge(b2, c2, "leads")

    conf_1 = match_graphs([g1, g2], embedder, max_iter=1)
    conf_30 = match_graphs([g1, g2], embedder, max_iter=30)

    # The far-end pair (Alpha/Alpha) should benefit from more iterations
    assert conf_30[(a1.id, a2.id)] >= conf_1[(a1.id, a2.id)]


# ---------------------------------------------------------------------------
# Confidence gate isolation
# ---------------------------------------------------------------------------


def test_confidence_gate_blocks_weak_anchor(embedder):
    """When an anchor's name similarity falls below the confidence gate,
    it should not contribute structural evidence.

    'Meridian Tech' / 'Meridian Technologies' have name sim ~0.85.
    With confidence_gate=0.95, the anchor is blocked → no propagation."""
    g1 = Graph(id="g1")
    apple1 = g1.add_entity("Apple")
    meridian1 = g1.add_entity("Meridian Technologies")
    g1.add_edge(apple1, meridian1, "acquired")

    g2 = Graph(id="g2")
    apple2 = g2.add_entity("Apple")
    meridian2 = g2.add_entity("Meridian Tech")
    g2.add_edge(apple2, meridian2, "acquired")

    conf_high_gate = match_graphs([g1, g2], embedder, confidence_gate=0.95)
    conf_low_gate = match_graphs([g1, g2], embedder, confidence_gate=0.5)

    # With high gate, Apple pair gets no structural boost from Meridian anchor
    assert (
        conf_high_gate[(apple1.id, apple2.id)] < conf_low_gate[(apple1.id, apple2.id)]
    )


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_confidence_is_monotonically_nondecreasing(embedder):
    """Confidence should never decrease as more iterations run."""
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault Inc")
    ceo1 = g1.add_entity("Elena Vasquez")
    g1.add_edge(meridian1, dv1, "acquired")
    g1.add_edge(meridian1, ceo1, "hired")

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault Inc")
    ceo2 = g2.add_entity("Elena Vasquez")
    g2.add_edge(meridian2, dv2, "purchased")
    g2.add_edge(meridian2, ceo2, "employed")

    prev_conf = match_graphs([g1, g2], embedder, max_iter=1)
    for n_iter in [2, 5, 10]:
        curr_conf = match_graphs([g1, g2], embedder, max_iter=n_iter)
        for pair, val in curr_conf.items():
            assert val >= prev_conf[pair] - 1e-9, (
                f"Confidence decreased for {pair}: {prev_conf[pair]:.6f} → {val:.6f} "
                f"at max_iter={n_iter}"
            )
        prev_conf = curr_conf
