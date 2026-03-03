"""Layer 2 tests for similarity propagation.

These tests exercise the propagate() function on small graphs with real
embeddings.  Tests verify that:

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

import numpy as np

from worldgraph.match import (
    Graph,
    LiteralNode,
    compute_functionality,
    propagate,
    select_matches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_ids(graph: Graph) -> list[str]:
    """Return only entity (non-literal) node IDs from a graph."""
    return [nid for nid, n in graph.nodes.items() if not isinstance(n, LiteralNode)]


def run_propagation(
    graph_a: Graph,
    graph_b: Graph,
    embed_relation,
    literal_embeddings: dict[str, np.ndarray],
    relations: list[str],
    threshold: float = 0.8,
    **kwargs,
):
    """Convenience wrapper: embed relations, compute functionality, propagate."""
    rel_embs = {r: embed_relation(r) for r in relations}
    rel_embs["is named"] = embed_relation("is named")
    func = compute_functionality([graph_a, graph_b], rel_embs, threshold)
    return propagate(
        graph_a,
        graph_b,
        literal_embeddings,
        rel_embs,
        func,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Correct matches
# ---------------------------------------------------------------------------


def test_matching_names_and_relations_produce_matches(embed, embed_relation):
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

    lit_embs = embed(["Apple", "Beats"])

    confidence = run_propagation(g1, g2, embed_relation, lit_embs, ["acquired"])

    assert confidence[(apple1.id, apple2.id)] > 0.8
    assert confidence[(beats1.id, beats2.id)] > 0.8

    assert confidence[(apple1.id, beats2.id)] < 0.5
    assert confidence[(beats1.id, apple2.id)] < 0.5


def test_synonym_relations_propagate(embed, embed_relation):
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

    lit_embs = embed(["Apple", "Beats"])

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["acquired", "purchased"]
    )

    assert confidence[(apple1.id, apple2.id)] > 0.8
    assert confidence[(beats1.id, beats2.id)] > 0.8


# ---------------------------------------------------------------------------
# No spurious matches (GH issue #1)
# ---------------------------------------------------------------------------


def test_dissimilar_relations_do_not_propagate(embed, embed_relation):
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

    lit_embs = embed(["Apple", "Beats", "Tokyo", "Japan"])

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["acquired", "located in"]
    )

    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )
    assert matches == [], f"Spurious matches found: {matches}"


def test_weak_neighbors_do_not_produce_matches(embed, embed_relation):
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

    lit_embs = embed(["Apple", "Beats", "Google", "YouTube"])

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["acquired"]
    )

    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )
    assert matches == [], f"Spurious matches from weak neighbors: {matches}"


def test_many_weak_paths_do_not_accumulate(embed, embed_relation):
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

    lit_embs = embed(
        ["Org", "Target", "Project", "Person", "City", "Country", "River", "Event"],
    )

    relations = ["acquired", "funded", "hired", "located in", "borders", "hosts"]
    confidence = run_propagation(g1, g2, embed_relation, lit_embs, relations)

    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )
    assert matches == [], f"Spurious matches from accumulated weak paths: {matches}"


# ---------------------------------------------------------------------------
# Incoming edges
# ---------------------------------------------------------------------------


def test_incoming_edges_propagate(embed, embed_relation):
    """Structural evidence should propagate through incoming edges, not just
    outgoing.

    DataVault has identical names (high literal sim). The incoming edge should
    propagate that confidence to the Meridian pair via exponential sum."""
    g1 = Graph(id="g1")
    meridian1 = g1.add_entity("Meridian Technologies")
    dv1 = g1.add_entity("DataVault")
    g1.add_edge(meridian1, dv1, "acquired")

    g2 = Graph(id="g2")
    meridian2 = g2.add_entity("Meridian Tech")
    dv2 = g2.add_entity("DataVault")
    g2.add_edge(meridian2, dv2, "purchased")

    lit_embs = embed(["Meridian Technologies", "Meridian Tech", "DataVault"])

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["acquired", "purchased"]
    )

    # With only name similarity and no structure, entity-entity confidence
    # would be 0 (entities start at 0). Any positive value means structural
    # evidence propagated.
    assert confidence[(meridian1.id, meridian2.id)] > 0, (
        "Incoming edge path did not propagate evidence to Meridian pair"
    )


# ---------------------------------------------------------------------------
# Functionality weighting
# ---------------------------------------------------------------------------


def test_functional_relation_produces_stronger_evidence(embed, embed_relation):
    """A 1:1 (functional) relation should produce stronger confidence boost
    than a many-to-one relation, all else being equal."""
    lit_embs = embed([
        "Meridian Technologies", "Meridian Tech", "DataVault",
        "Apple", "Google", "Microsoft", "Beats", "YouTube",
    ])

    # Background graphs for functionality: 'acquired' is 1:1, 'invested in' has fan-in
    fg1 = Graph(id="fg1")
    fg1_apple = fg1.add_entity("Apple")
    fg1_beats = fg1.add_entity("Beats")
    fg1_google = fg1.add_entity("Google")
    fg1_yt = fg1.add_entity("YouTube")
    fg1.add_edge(fg1_apple, fg1_beats, "acquired")
    fg1.add_edge(fg1_google, fg1_yt, "acquired")

    fg2 = Graph(id="fg2")
    fg2_apple = fg2.add_entity("Apple")
    fg2_google = fg2.add_entity("Google")
    fg2_ms = fg2.add_entity("Microsoft")
    fg2_dv = fg2.add_entity("DataVault")
    fg2.add_edge(fg2_apple, fg2_dv, "invested in")
    fg2.add_edge(fg2_google, fg2_dv, "invested in")
    fg2.add_edge(fg2_ms, fg2_dv, "invested in")

    rel_embs = {
        "acquired": embed_relation("acquired"),
        "invested in": embed_relation("invested in"),
        "is named": embed_relation("is named"),
    }
    func = compute_functionality([fg1, fg2], rel_embs)
    assert func["acquired"].inverse > func["invested in"].inverse

    # Propagation with 'acquired' (high inverse functionality)
    g1a = Graph(id="g1a")
    m1a = g1a.add_entity("Meridian Technologies")
    dv1a = g1a.add_entity("DataVault")
    g1a.add_edge(m1a, dv1a, "acquired")

    g2a = Graph(id="g2a")
    m2a = g2a.add_entity("Meridian Tech")
    dv2a = g2a.add_entity("DataVault")
    g2a.add_edge(m2a, dv2a, "acquired")

    confidence_acq = propagate(g1a, g2a, lit_embs, rel_embs, func)

    # Propagation with 'invested in' (low inverse functionality)
    g1i = Graph(id="g1i")
    m1i = g1i.add_entity("Meridian Technologies")
    dv1i = g1i.add_entity("DataVault")
    g1i.add_edge(m1i, dv1i, "invested in")

    g2i = Graph(id="g2i")
    m2i = g2i.add_entity("Meridian Tech")
    dv2i = g2i.add_entity("DataVault")
    g2i.add_edge(m2i, dv2i, "invested in")

    confidence_inv = propagate(g1i, g2i, lit_embs, rel_embs, func)

    assert confidence_acq[(m1a.id, m2a.id)] > confidence_inv[(m1i.id, m2i.id)], (
        f"Functional relation confidence ({confidence_acq[(m1a.id, m2a.id)]}) should exceed "
        f"non-functional ({confidence_inv[(m1i.id, m2i.id)]})"
    )


# ---------------------------------------------------------------------------
# Multi-hop propagation
# ---------------------------------------------------------------------------


def test_multi_hop_propagation_across_iterations(embed, embed_relation):
    """Evidence propagates through a chain: a high-confidence anchor at
    the end boosts intermediate nodes, which in turn boost further nodes.

    James Chen pair has identical names → high literal sim.
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

    lit_embs = embed([
        "Meridian Technologies", "Meridian Tech",
        "Alpha Corp", "Beta Inc", "James Chen",
    ])

    relations = ["acquired", "purchased", "founded by"]
    rel_embs = {r: embed_relation(r) for r in relations}
    rel_embs["is named"] = embed_relation("is named")
    func = compute_functionality([g1, g2], rel_embs)

    confidence = propagate(g1, g2, lit_embs, rel_embs, func)

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


def test_name_variation_with_structural_reinforcement(embed, embed_relation):
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

    lit_embs = embed(
        ["Meridian Technologies", "Meridian Tech", "DataVault Inc", "Elena Vasquez"]
    )

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["acquired", "purchased", "hired", "employed"]
    )

    # Both should pass select_matches
    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )
    matched_pairs = {(a, b) for a, b in matches}
    assert (meridian1.id, meridian2.id) in matched_pairs
    assert (dv1.id, dv2.id) in matched_pairs


# ---------------------------------------------------------------------------
# Dangling entities
# ---------------------------------------------------------------------------


def test_dangling_entities_get_no_boost(embed, embed_relation):
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

    lit_embs = embed(["Apple", "Beats", "SolarGrid", "WindPower"])

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["acquired", "purchased", "hired"]
    )

    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )
    matched_ids = {eid for pair in matches for eid in pair}
    assert solar.id not in matched_ids
    assert wind.id not in matched_ids


# ---------------------------------------------------------------------------
# Exp-sum accumulation (bidirectional > unidirectional)
# ---------------------------------------------------------------------------


def test_bidirectional_edges_accumulate(embed, embed_relation):
    """Entity pairs connected by edges in both directions should accumulate
    evidence, producing higher confidence than a single edge."""
    lit_embs = embed(["Meridian Technologies", "Meridian Tech", "DataVault"])

    # Unidirectional
    g1u = Graph(id="g1u")
    m1u = g1u.add_entity("Meridian Technologies")
    dv1u = g1u.add_entity("DataVault")
    g1u.add_edge(m1u, dv1u, "acquired")

    g2u = Graph(id="g2u")
    m2u = g2u.add_entity("Meridian Tech")
    dv2u = g2u.add_entity("DataVault")
    g2u.add_edge(m2u, dv2u, "acquired")

    confidence_uni = run_propagation(
        g1u, g2u, embed_relation, lit_embs, ["acquired"]
    )

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

    confidence_bi = run_propagation(
        g1b, g2b, embed_relation, lit_embs, ["acquired", "subsidiary of"]
    )

    assert confidence_bi[(m1b.id, m2b.id)] >= confidence_uni[(m1u.id, m2u.id)], (
        f"Bidirectional ({confidence_bi[(m1b.id, m2b.id)]}) should be >= "
        f"unidirectional ({confidence_uni[(m1u.id, m2u.id)]})"
    )


# ---------------------------------------------------------------------------
# Structural override of name dissimilarity
# ---------------------------------------------------------------------------


def test_shared_anchor_does_not_override_name_dissimilarity(embed, embed_relation):
    """A shared high-confidence anchor should not cause entities with
    different names to match.

    NovaTech Labs has identical literal sim across graphs. "founded"
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
    for i, (person, org) in enumerate([
        ("Marcus Webb", "Alpha Corp"),
        ("Sarah Chen", "Beta Inc"),
        ("James Xu", "Gamma LLC"),
    ]):
        bg = Graph(id=f"bg{i}")
        p = bg.add_entity(person)
        o = bg.add_entity(org)
        bg.add_edge(p, o, "founded")
        bg_graphs.append(bg)

    lit_embs = embed(["Dr. Priya Sharma", "Dr. Elena Vasquez", "NovaTech Labs"])

    rel_embs = {
        "founded": embed_relation("founded"),
        "is named": embed_relation("is named"),
    }
    func = compute_functionality([g1, g2] + bg_graphs, rel_embs)

    # Premise: name similarity alone is below threshold
    sv_name_sim = float(np.dot(
        lit_embs["Dr. Priya Sharma"], lit_embs["Dr. Elena Vasquez"]
    ))
    assert sv_name_sim < 0.8

    confidence = propagate(g1, g2, lit_embs, rel_embs, func)

    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )

    # Neither NovaTech nor Sharma/Vasquez should match
    assert (nova1.id, nova2.id) not in matches, (
        "Identical names with no structural corroboration were incorrectly matched"
    )
    assert (sharma.id, vasquez.id) not in matches, (
        "Structural evidence from shared anchor overrode name dissimilarity"
    )


def test_similar_names_disjoint_neighborhoods_no_match(embed, embed_relation):
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

    lit_embs = embed([
        "Dr. Elena Vasquez", "Dr. Lena Vasquez",
        "Volta Systems", "Halcyon Genomics",
    ])

    # Premise: names are similar enough that old system would match them
    name_sim = float(np.dot(
        lit_embs["Dr. Elena Vasquez"], lit_embs["Dr. Lena Vasquez"]
    ))
    assert name_sim >= 0.8, (
        f"Premise failed: Elena/Lena name_sim ({name_sim:.3f}) below threshold"
    )

    # Premise: neighbor names have no similarity
    nbr_sim = float(np.dot(
        lit_embs["Volta Systems"], lit_embs["Halcyon Genomics"]
    ))
    assert nbr_sim < 0.5

    confidence = run_propagation(
        g1, g2, embed_relation, lit_embs, ["is CEO of"]
    )

    matches = select_matches(
        confidence, _entity_ids(g1), _entity_ids(g2), threshold=0.8
    )
    matched_pairs = {(a, b) for a, b in matches}

    assert (elena.id, lena.id) not in matched_pairs, (
        f"Similar names with disjoint neighborhoods were incorrectly matched "
        f"(name_sim={name_sim:.3f}, no structural support)"
    )
