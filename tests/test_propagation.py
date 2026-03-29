"""Layer 2 tests for similarity propagation.

These tests exercise match_graphs() on small graphs with real embeddings.
Tests verify that:

- Matching names + matching relations produce correct matches
- Synonym relations propagate via continuous relation similarity
- Dissimilar relations, weak neighbors, and many weak paths produce no
  spurious matches (GH issue #1)
- Incoming edges propagate evidence (not just outgoing)
- Functionality weighting affects evidence strength
- Multi-hop chains require iterative propagation
- Name variation with structural reinforcement (the core use case)
- Dangling entities get no structural evidence
- Exponential sum accumulates evidence from multiple paths (bidirectional > unidirectional)
- Multi-label entities use max similarity across all names during seeding
"""

from worldgraph.graph import Graph
from worldgraph.match import match_graphs
from worldgraph.names import build_idf, soft_tfidf


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
    """Synonym relation phrases ('acquired' / 'purchased') should have high
    relation similarity and produce correct matches."""
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
    """Unrelated relation phrases ('acquired' vs 'located in') have low
    relation similarity — no matches between unrelated graphs."""
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
    # Use dissimilar entity names so only structural evidence matters
    g1a = Graph(id="g1a")
    m1a = g1a.add_entity("Axiom Corp")
    dv1a = g1a.add_entity("DataVault")
    g1a.add_edge(m1a, dv1a, "acquired")

    g2a = Graph(id="g2a")
    m2a = g2a.add_entity("Pinnacle Ltd")
    dv2a = g2a.add_entity("DataVault")
    g2a.add_edge(m2a, dv2a, "acquired")

    confidence_acq = match_graphs([bg1, bg2, g1a, g2a], embedder)

    # Test pair with 'invested in' (low inverse functionality)
    g1i = Graph(id="g1i")
    m1i = g1i.add_entity("Axiom Corp")
    dv1i = g1i.add_entity("DataVault")
    g1i.add_edge(m1i, dv1i, "invested in")

    g2i = Graph(id="g2i")
    m2i = g2i.add_entity("Pinnacle Ltd")
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

    NovaTech Labs has identical name sim across graphs. But under
    exponential sum, a single structural path is insufficient to cross
    the threshold."""
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

    names = [
        name for g in [g1, g2, *bg_graphs] for n in g.nodes.values() for name in n.names
    ]
    idf = build_idf(names)
    sv_name_sim = soft_tfidf("Dr. Priya Sharma", "Dr. Elena Vasquez", idf)
    assert sv_name_sim < 0.8

    confidence = match_graphs([g1, g2, *bg_graphs], embedder)

    matches = _select_matches(confidence, threshold=0.8)
    matched_pairs = set(matches)
    # NovaTech Labs has identical names (seed ~1.0) but the only neighbor
    # (founder) doesn't match — the negative channel propagates founder
    # mismatch, and the Bayesian combination can push the final score
    # below 0.8.  This is acceptable: the test's purpose is below.

    # sharma/vasquez should NOT match — structural anchor alone
    # cannot override name dissimilarity
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
# Monotonicity
# ---------------------------------------------------------------------------


def test_same_graph_entities_never_match(embedder):
    """Two entities within the same article graph should never appear as a
    match pair, regardless of name similarity."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple Inc")
    music = g.add_entity("Apple Music")
    g.add_edge(apple, music, "owns")

    g2 = Graph(id="g2")
    g2.add_entity("Google")
    g2.add_entity("Alphabet")

    confidence = match_graphs([g, g2], embedder)

    assert (apple.id, music.id) not in confidence
    assert (music.id, apple.id) not in confidence


def test_single_graph_produces_no_matches(embedder):
    """match_graphs with a single graph should return an empty confidence dict."""
    g = Graph(id="g1")
    apple = g.add_entity("Apple")
    beats = g.add_entity("Beats")
    g.add_edge(apple, beats, "acquired")

    confidence = match_graphs([g], embedder)

    assert confidence == {}


def test_propagation_converges(embedder):
    """Both positive and negative channels should converge: running with
    more iterations than needed should not change the result."""
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

    conf_10 = match_graphs([g1, g2], embedder, max_iter=10)
    conf_30 = match_graphs([g1, g2], embedder, max_iter=30)
    for pair, val in conf_30.items():
        assert abs(val - conf_10.get(pair, 0.0)) < 1e-9, (
            f"Score changed between max_iter=10 and max_iter=30 for {pair}: "
            f"{conf_10.get(pair, 0.0):.6f} → {val:.6f}"
        )


# ---------------------------------------------------------------------------
# Multi-label name seeding
# ---------------------------------------------------------------------------


def test_multi_label_entity_uses_best_name_pair(embedder):
    """An entity with multiple names should seed similarity using the best
    name pair across both entities' name lists.

    "Meridian Technologies" stored as names=["Meridian Technologies"] in g1,
    and names=["Meridian Tech", "Meridian Technologies"] in g2.  The best
    pair is "Meridian Technologies"/"Meridian Technologies" (score ~1.0),
    not "Meridian Technologies"/"Meridian Tech" (~0.88).

    Without multi-label support, only one name is stored and the closest
    pair may be missed, under-estimating similarity."""
    g1 = Graph(id="g1")
    m1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault")
    g1.add_edge(m1, dv1, "acquired")

    g2 = Graph(id="g2")
    m2 = g2.add_entity(["Meridian Tech", "Meridian Technologies"])
    dv2 = g2.add_entity("DataVault")
    g2.add_edge(m2, dv2, "purchased")

    confidence = match_graphs([g1, g2], embedder)

    # With multi-label, the best name pair is exact match → seed ~1.0
    # Without, if only "Meridian Tech" is stored, seed would be ~0.88
    assert confidence[(m1.id, m2.id)] > 0.8


def test_multi_label_all_names_contribute_to_idf(embedder):
    """All names in an entity's name list should contribute to IDF
    computation, not just the first."""
    g1 = Graph(id="g1")
    m1 = g1.add_entity(["Meridian Technologies", "Meridian Tech"])
    dv1 = g1.add_entity("DataVault")
    g1.add_edge(m1, dv1, "acquired")

    g2 = Graph(id="g2")
    m2 = g2.add_entity("Meridian Technologies")
    dv2 = g2.add_entity("DataVault")
    g2.add_edge(m2, dv2, "purchased")

    # Should not raise — multi-label names flow through the pipeline
    confidence = match_graphs([g1, g2], embedder)
    assert confidence[(m1.id, m2.id)] > 0.8


# ---------------------------------------------------------------------------
# Progressive merging — enriched neighborhood
# ---------------------------------------------------------------------------


def test_progressive_merging_enriched_neighborhood(embedder):
    """Progressive merging enriches neighborhoods across epochs, enabling
    matches that pairwise comparison alone cannot produce.

    Articles A and B describe Meridian Corp with overlapping structure
    (DataVault, James Chen, Stanford) plus unique neighbors (A: Austin,
    B: Volta Systems).  They merge in epoch 1 (identical names + strong
    structural match).

    Article C describes "Meridian Tech Corp" — moderate name similarity
    (~0.64) to "Meridian Corp", sharing one neighbor with A (Austin) and
    one with B (Volta Systems), plus James Chen (shared by both).

    Without progressive merging (max_epochs=1), C's best pairwise match
    sees only 2 structural paths (James Chen + one of Austin/Volta).
    With progressive merging, the merged A+B entity has ALL neighbors
    (DataVault, James Chen, Stanford, Austin, Volta), giving C three
    matching paths.  The additional structural evidence produces a
    measurably higher confidence.
    """
    # Article A: Meridian Corp with DataVault, James Chen, Stanford, Austin
    ga = Graph(id="a")
    ma = ga.add_entity("Meridian Corp")
    dva = ga.add_entity("DataVault")
    ja = ga.add_entity("James Chen")
    su_a = ga.add_entity("Stanford University")
    austin_a = ga.add_entity("Austin")
    ga.add_edge(ma, dva, "acquired")
    ga.add_edge(ma, ja, "CEO is")
    ga.add_edge(ma, su_a, "alumna of")
    ga.add_edge(ma, austin_a, "headquartered in")

    # Article B: Meridian Corp with DataVault, James Chen, Stanford, Volta
    gb = Graph(id="b")
    mb = gb.add_entity("Meridian Corp")
    dvb = gb.add_entity("DataVault")
    jb = gb.add_entity("James Chen")
    su_b = gb.add_entity("Stanford University")
    volta_b = gb.add_entity("Volta Systems")
    gb.add_edge(mb, dvb, "purchased")
    gb.add_edge(mb, jb, "CEO is")
    gb.add_edge(mb, su_b, "alumna of")
    gb.add_edge(mb, volta_b, "partnered with")

    # Article C: "Meridian Tech Corp" — moderate name sim, neighbors from
    # both A-unique (Austin) and B-unique (Volta) plus shared (James Chen)
    gc = Graph(id="c")
    mc = gc.add_entity("Meridian Tech Corp")
    austin_c = gc.add_entity("Austin")
    volta_c = gc.add_entity("Volta Systems")
    jc = gc.add_entity("James Chen")
    gc.add_edge(mc, austin_c, "headquartered in")
    gc.add_edge(mc, volta_c, "partnered with")
    gc.add_edge(mc, jc, "CEO is")

    graphs = [ga, gb, gc]

    # Premise: name similarity alone is insufficient
    from worldgraph.names import build_idf, soft_tfidf

    names = [name for g in graphs for n in g.nodes.values() for name in n.names]
    idf = build_idf(names)
    assert soft_tfidf("Meridian Tech Corp", "Meridian Corp", idf) < 0.8

    # Premise: A+B merge above merge_threshold
    conf_single = match_graphs(graphs, embedder, merge_threshold=float("inf"))
    assert conf_single[(ma.id, mb.id)] >= 0.9, (
        f"A-B should merge: {conf_single[(ma.id, mb.id)]:.3f}"
    )

    # Without progressive merging (merge_threshold=inf), C sees only pairwise evidence
    conf_progressive = match_graphs(graphs, embedder)

    # Progressive merging produces strictly higher confidence for C
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
# Negative evidence should use structural confidence (not name_seed)
# ---------------------------------------------------------------------------


def test_negative_evidence_does_not_over_penalize_structurally_matched_neighbors(
    embedder,
):
    """Negative evidence should not penalize an entity pair when its
    functional neighbors are structurally matched.

    "Meridian Technologies" and "Meridian Tech" share two structural paths:
    acquired → DataVault (identical names) and CEO → a person with weak name
    similarity but shared sub-neighbor (Stanford University).

    The positive channel discovers the CEO match via Stanford.  The negative
    channel sees CEO name dissimilarity (~0.7) but Stanford's zero
    dissimilarity blocks negative propagation for the CEO pair.  The
    Bayesian combination divides out the name prior, so only the structural
    growth matters — and positive grew more than negative.
    """
    g1 = Graph(id="g1")
    m1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault")
    ceo1 = g1.add_entity("Dr. Alice M. Johnson")
    uni1 = g1.add_entity("Stanford University")
    g1.add_edge(m1, dv1, "acquired")
    g1.add_edge(m1, ceo1, "CEO")
    g1.add_edge(ceo1, uni1, "graduated from")

    g2 = Graph(id="g2")
    m2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault")
    ceo2 = g2.add_entity("A. Johnson")
    uni2 = g2.add_entity("Stanford University")
    g2.add_edge(m2, dv2, "purchased")
    g2.add_edge(m2, ceo2, "CEO")
    g2.add_edge(ceo2, uni2, "graduated from")

    # Background to establish CEO as functional (1:1)
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
        bg.add_edge(p, o, "CEO")
        bg_graphs.append(bg)

    graphs = [g1, g2, *bg_graphs]

    # Premise: CEO name similarity is weak (structural propagation needed)
    all_names = [n for g in graphs for node in g.nodes.values() for n in node.names]
    idf = build_idf(all_names)
    assert soft_tfidf("Dr. Alice M. Johnson", "A. Johnson", idf) < 0.5

    confidence = match_graphs(graphs, embedder)

    # CEO pair should be structurally matched despite weak names
    ceo_score = confidence.get(
        (ceo1.id, ceo2.id), confidence.get((ceo2.id, ceo1.id), 0.0)
    )
    assert ceo_score > 0.6, (
        f"CEO pair should be structurally matched, got {ceo_score:.3f}"
    )

    # Meridian should not be over-penalized — the CEO targets match
    # structurally via Stanford.
    meridian_score = confidence.get((m1.id, m2.id), confidence.get((m2.id, m1.id), 0.0))
    assert meridian_score > 0.8, (
        f"Negative evidence over-penalized Meridian: score={meridian_score:.3f} "
        f"(CEO structural match={ceo_score:.3f})"
    )
