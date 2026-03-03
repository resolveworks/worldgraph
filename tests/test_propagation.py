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
    Entity,
    Edge,
    Graph,
    compute_functionality,
    propagate,
    select_matches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph(graph_id: str, edges: list[tuple[str, str, str]]) -> Graph:
    """Build a Graph from (src_name, tgt_name, relation) triples."""
    entities: dict[str, Entity] = {}
    edge_objs: list[Edge] = []
    for src_name, tgt_name, relation in edges:
        src_id = f"{graph_id}:{src_name}"
        tgt_id = f"{graph_id}:{tgt_name}"
        if src_id not in entities:
            entities[src_id] = Entity(id=src_id, name=src_name, graph_id=graph_id)
        if tgt_id not in entities:
            entities[tgt_id] = Entity(id=tgt_id, name=tgt_name, graph_id=graph_id)
        edge_objs.append(Edge(source=src_id, target=tgt_id, relation=relation))
    g = Graph(id=graph_id, entities=entities, edges=edge_objs)
    return g


def compute_seeds(
    graph_a: Graph,
    graph_b: Graph,
    name_embeddings: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Compute the name-similarity seed for every entity pair."""
    seeds: dict[tuple[str, str], float] = {}
    for eid_a in graph_a.entities:
        name_a = graph_a.entities[eid_a].name
        emb_a = name_embeddings.get(name_a)
        for eid_b in graph_b.entities:
            name_b = graph_b.entities[eid_b].name
            emb_b = name_embeddings.get(name_b)
            if emb_a is not None and emb_b is not None:
                seeds[(eid_a, eid_b)] = max(0.0, float(np.dot(emb_a, emb_b)))
            else:
                seeds[(eid_a, eid_b)] = 0.0
    return seeds


def run_propagation(
    graph_a: Graph,
    graph_b: Graph,
    embed_relation,
    name_embeddings: dict[str, np.ndarray],
    relations: list[str],
    threshold: float = 0.8,
    **kwargs,
):
    """Convenience wrapper: embed relations, compute functionality, propagate.

    Returns (confidence, seeds) where seeds is the name-similarity baseline.
    """
    rel_embs = {r: embed_relation(r) for r in relations}
    func = compute_functionality([graph_a, graph_b], rel_embs, threshold)
    confidence = propagate(
        graph_a,
        graph_b,
        name_embeddings,
        rel_embs,
        func,
        **kwargs,
    )
    seeds = compute_seeds(graph_a, graph_b, name_embeddings)
    return confidence, seeds


# ---------------------------------------------------------------------------
# Correct matches
# ---------------------------------------------------------------------------


def test_matching_names_and_relations_produce_matches(embed, embed_relation):
    """Two graphs with the same entity names and identical relations should
    produce high-confidence matches for the correct entity pairs."""
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "acquired")])

    name_embs = embed(["Apple", "Beats"])

    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, ["acquired"]
    )

    # Correct pairs should have high confidence
    assert confidence[("g1:Apple", "g2:Apple")] > 0.9
    assert confidence[("g1:Beats", "g2:Beats")] > 0.9

    # Wrong pairs should have low confidence (just name_sim of unrelated words)
    assert confidence[("g1:Apple", "g2:Beats")] < 0.5
    assert confidence[("g1:Beats", "g2:Apple")] < 0.5


def test_synonym_relations_propagate(embed, embed_relation):
    """Synonym relation phrases ('acquired' / 'purchased') should pass the
    relation gate and produce correct matches."""
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "purchased")])

    name_embs = embed(["Apple", "Beats"])

    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, ["acquired", "purchased"]
    )

    assert confidence[("g1:Apple", "g2:Apple")] > 0.9
    assert confidence[("g1:Beats", "g2:Beats")] > 0.9


# ---------------------------------------------------------------------------
# No spurious matches (GH issue #1)
# ---------------------------------------------------------------------------


def test_dissimilar_relations_do_not_propagate(embed, embed_relation):
    """Unrelated relation phrases ('acquired' vs 'located in') should not
    pass the relation gate — no matches between unrelated graphs.

    g1: Apple --acquired--> Beats
    g2: Tokyo --located in--> Japan
    """
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Tokyo", "Japan", "located in")])

    name_embs = embed(["Apple", "Beats", "Tokyo", "Japan"])

    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, ["acquired", "located in"]
    )

    # No matches
    matches = select_matches(
        confidence, list(g1.entities), list(g2.entities), threshold=0.8
    )
    assert matches == [], f"Spurious matches found: {matches}"


def test_weak_neighbors_do_not_produce_matches(embed, embed_relation):
    """Even with identical relation phrases, weak neighbor similarity
    should not produce matches.  The exponential sum gives weak paths
    diminishing returns — they contribute small amounts but never enough
    to cross the match threshold.

    g1: Apple --acquired--> Beats
    g2: Google --acquired--> YouTube
    """
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Google", "YouTube", "acquired")])

    name_embs = embed(["Apple", "Beats", "Google", "YouTube"])

    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, ["acquired"]
    )

    matches = select_matches(
        confidence, list(g1.entities), list(g2.entities), threshold=0.8
    )
    assert matches == [], f"Spurious matches from weak neighbors: {matches}"


def test_many_weak_paths_do_not_accumulate(embed, embed_relation):
    """Many unrelated edges should not produce matches.

    g1: Org --acquired--> Target, Org --funded--> Project, Org --hired--> Person
    g2: City --located in--> Country, City --borders--> River, City --hosts--> Event
    """
    g1 = make_graph(
        "g1",
        [
            ("Org", "Target", "acquired"),
            ("Org", "Project", "funded"),
            ("Org", "Person", "hired"),
        ],
    )
    g2 = make_graph(
        "g2",
        [
            ("City", "Country", "located in"),
            ("City", "River", "borders"),
            ("City", "Event", "hosts"),
        ],
    )

    name_embs = embed(
        ["Org", "Target", "Project", "Person", "City", "Country", "River", "Event"],
    )

    relations = ["acquired", "funded", "hired", "located in", "borders", "hosts"]
    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, relations
    )

    matches = select_matches(
        confidence, list(g1.entities), list(g2.entities), threshold=0.8
    )
    assert matches == [], f"Spurious matches from accumulated weak paths: {matches}"


# ---------------------------------------------------------------------------
# Incoming edges
# ---------------------------------------------------------------------------


def test_incoming_edges_propagate(embed, embed_relation):
    """Structural evidence should propagate through incoming edges, not just
    outgoing.

    g1: Meridian Technologies --acquired--> DataVault
    g2: Meridian Tech          --purchased--> DataVault

    DataVault has identical names (high seed). The incoming edge should
    propagate that confidence to the Meridian pair, boosting it above
    what name similarity alone gives.
    """
    g1 = make_graph("g1", [("Meridian Technologies", "DataVault", "acquired")])
    g2 = make_graph("g2", [("Meridian Tech", "DataVault", "purchased")])

    name_embs = embed(
        ["Meridian Technologies", "Meridian Tech", "DataVault"]
    )

    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, ["acquired", "purchased"]
    )

    # Meridian pair should be boosted above its name-similarity seed
    mt_pair = ("g1:Meridian Technologies", "g2:Meridian Tech")
    assert confidence[mt_pair] > seeds[mt_pair], (
        "Incoming edge path did not boost Meridian pair above name-similarity seed"
    )


# ---------------------------------------------------------------------------
# Functionality weighting
# ---------------------------------------------------------------------------


def test_functional_relation_produces_stronger_evidence(embed, embed_relation):
    """A 1:1 (functional) relation should produce stronger confidence boost
    than a many-to-one relation, all else being equal.

    Both use name variations so propagation is observable.
    The propagation path is src --r--> tgt, which uses inverse functionality
    (how unique is the source given the target?).

    'acquired' is 1:1 → inverse functionality ≈ 1.0.
    'invested in' has fan-in (many sources → same target) → inverse func < 1.0.
    """
    name_embs = embed(
        [
            "Meridian Technologies",
            "Meridian Tech",
            "DataVault",
            "Apple",
            "Google",
            "Microsoft",
            "Beats",
            "YouTube",
        ],
    )

    # Build functionality: 'acquired' is 1:1, 'invested in' has fan-in
    func_graphs = [
        make_graph(
            "fg1",
            [
                ("Apple", "Beats", "acquired"),
                ("Google", "YouTube", "acquired"),
            ],
        ),
        make_graph(
            "fg2",
            [
                ("Apple", "DataVault", "invested in"),
                ("Google", "DataVault", "invested in"),
                ("Microsoft", "DataVault", "invested in"),
            ],
        ),
    ]
    rel_embs = {
        "acquired": embed_relation("acquired"),
        "invested in": embed_relation("invested in"),
    }
    func = compute_functionality(func_graphs, rel_embs)
    assert func["acquired"].inverse > func["invested in"].inverse, (
        f"Premise failed: acquired inv_func={func['acquired'].inverse}, "
        f"invested in inv_func={func['invested in'].inverse}"
    )

    # Run propagation with 'acquired' edges (high inverse functionality)
    g1_acq = make_graph(
        "g1a", [("Meridian Technologies", "DataVault", "acquired")]
    )
    g2_acq = make_graph("g2a", [("Meridian Tech", "DataVault", "acquired")])
    confidence_acq = propagate(g1_acq, g2_acq, name_embs, rel_embs, func)

    # Run propagation with 'invested in' edges (low inverse functionality)
    g1_inv = make_graph(
        "g1i", [("Meridian Technologies", "DataVault", "invested in")]
    )
    g2_inv = make_graph(
        "g2i", [("Meridian Tech", "DataVault", "invested in")]
    )
    confidence_inv = propagate(g1_inv, g2_inv, name_embs, rel_embs, func)

    # The functional relation should produce higher confidence
    mt_acq = ("g1a:Meridian Technologies", "g2a:Meridian Tech")
    mt_inv = ("g1i:Meridian Technologies", "g2i:Meridian Tech")
    assert confidence_acq[mt_acq] > confidence_inv[mt_inv], (
        f"Functional relation confidence ({confidence_acq[mt_acq]}) should exceed "
        f"non-functional ({confidence_inv[mt_inv]})"
    )


# ---------------------------------------------------------------------------
# Multi-hop propagation
# ---------------------------------------------------------------------------


def test_multi_hop_propagation_across_iterations(embed, embed_relation):
    """Evidence propagates through a chain: a high-confidence anchor at
    the end boosts intermediate nodes, which in turn boost further nodes.

    g1: Meridian Technologies --acquired--> Alpha Corp --founded by--> James Chen
    g2: Meridian Tech         --purchased--> Beta Inc   --founded by--> James Chen

    James Chen pair has seed ≈ 1.0 (identical names).
    Alpha Corp / Beta Inc have very low name similarity — they can only be
    matched through their shared, confidently-matched neighbor (James Chen).
    Meridian Technologies / Meridian Tech then benefit from the chain through
    the now-matched Alpha Corp / Beta Inc.
    """
    g1 = make_graph(
        "g1",
        [
            ("Meridian Technologies", "Alpha Corp", "acquired"),
            ("Alpha Corp", "James Chen", "founded by"),
        ],
    )
    g2 = make_graph(
        "g2",
        [
            ("Meridian Tech", "Beta Inc", "purchased"),
            ("Beta Inc", "James Chen", "founded by"),
        ],
    )

    name_embs = embed(
        ["Meridian Technologies", "Meridian Tech", "Alpha Corp", "Beta Inc", "James Chen"],
    )

    relations = ["acquired", "purchased", "founded by"]
    rel_embs = {r: embed_relation(r) for r in relations}
    func = compute_functionality([g1, g2], rel_embs)

    seeds = compute_seeds(g1, g2, name_embs)
    mt_pair = ("g1:Meridian Technologies", "g2:Meridian Tech")
    ab_pair = ("g1:Alpha Corp", "g2:Beta Inc")

    # Verify premise: Alpha Corp / Beta Inc have name sim below confidence gate
    assert seeds[ab_pair] < 0.8, (
        f"Premise failed: Alpha Corp / Beta Inc seed too high ({seeds[ab_pair]})"
    )

    # After convergence: intermediate pair boosted by James Chen chain
    confidence = propagate(g1, g2, name_embs, rel_embs, func)
    assert confidence[ab_pair] > seeds[ab_pair], (
        "Alpha Corp / Beta Inc not boosted despite shared James Chen neighbor"
    )

    # Meridian pair boosted via the full chain
    assert confidence[mt_pair] > seeds[mt_pair], (
        "Multi-hop propagation failed: Meridian pair not boosted after convergence"
    )


# ---------------------------------------------------------------------------
# Name variation with structural reinforcement
# ---------------------------------------------------------------------------


def test_name_variation_with_structural_reinforcement(embed, embed_relation):
    """The core use case: similar-but-not-identical entity names get matched
    when structural evidence reinforces them.

    g1: Meridian Technologies --acquired--> DataVault Inc
    g2: Meridian Tech         --purchased--> DataVault Inc
    """
    g1 = make_graph("g1", [("Meridian Technologies", "DataVault Inc", "acquired")])
    g2 = make_graph("g2", [("Meridian Tech", "DataVault Inc", "purchased")])

    name_embs = embed(
        ["Meridian Technologies", "Meridian Tech", "DataVault Inc"]
    )

    relations = ["acquired", "purchased"]
    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, relations
    )

    # The varied-name pair should be boosted above its seed
    mt_pair = ("g1:Meridian Technologies", "g2:Meridian Tech")
    assert confidence[mt_pair] > seeds[mt_pair], (
        "Name variation pair got no structural reinforcement"
    )

    # Both should pass select_matches
    matches = select_matches(
        confidence,
        list(g1.entities),
        list(g2.entities),
        threshold=0.8,
    )
    matched_names = [(g1.entities[a].name, g2.entities[b].name) for a, b in matches]
    assert ("Meridian Technologies", "Meridian Tech") in matched_names
    assert ("DataVault Inc", "DataVault Inc") in matched_names


# ---------------------------------------------------------------------------
# Dangling entities
# ---------------------------------------------------------------------------


def test_dangling_entities_get_no_boost(embed, embed_relation):
    """Entities with no matching structure should not be boosted beyond
    their name-similarity seed, even when other entities match.

    g1: Apple --acquired--> Beats,   Apple --hired--> SolarGrid
    g2: Apple --purchased--> Beats,  Apple --hired--> WindPower

    Apple-Apple and Beats-Beats should match.  SolarGrid and WindPower are
    dangling — different names, no structural path to reinforce them.
    """
    g1 = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Apple", "SolarGrid", "hired"),
        ],
    )
    g2 = make_graph(
        "g2",
        [
            ("Apple", "Beats", "purchased"),
            ("Apple", "WindPower", "hired"),
        ],
    )

    name_embs = embed(["Apple", "Beats", "SolarGrid", "WindPower"])

    relations = ["acquired", "purchased", "hired"]
    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, relations
    )

    # select_matches should not include dangling entities
    matches = select_matches(
        confidence, list(g1.entities), list(g2.entities), threshold=0.8
    )
    matched_ids = {eid for pair in matches for eid in pair}
    assert "g1:SolarGrid" not in matched_ids
    assert "g2:WindPower" not in matched_ids


# ---------------------------------------------------------------------------
# Noisy-OR accumulation (bidirectional > unidirectional)
# ---------------------------------------------------------------------------


def test_bidirectional_edges_accumulate_via_noisy_or(embed, embed_relation):
    """Entity pairs connected by edges in both directions should accumulate
    evidence via noisy-OR, producing higher confidence than a single edge.

    Uses name variations so propagation is observable.

    Unidirectional:
        g1: Meridian Technologies --acquired--> DataVault
        g2: Meridian Tech         --acquired--> DataVault

    Bidirectional:
        g1: Meridian Technologies --acquired--> DataVault,  DataVault --subsidiary of--> Meridian Technologies
        g2: Meridian Tech         --acquired--> DataVault,  DataVault --subsidiary of--> Meridian Tech
    """
    name_embs = embed(
        ["Meridian Technologies", "Meridian Tech", "DataVault"]
    )

    # Unidirectional
    g1_uni = make_graph("g1u", [("Meridian Technologies", "DataVault", "acquired")])
    g2_uni = make_graph("g2u", [("Meridian Tech", "DataVault", "acquired")])

    relations_uni = ["acquired"]
    confidence_uni, _ = run_propagation(
        g1_uni, g2_uni, embed_relation, name_embs, relations_uni
    )

    # Bidirectional
    g1_bi = make_graph(
        "g1b",
        [
            ("Meridian Technologies", "DataVault", "acquired"),
            ("DataVault", "Meridian Technologies", "subsidiary of"),
        ],
    )
    g2_bi = make_graph(
        "g2b",
        [
            ("Meridian Tech", "DataVault", "acquired"),
            ("DataVault", "Meridian Tech", "subsidiary of"),
        ],
    )

    relations_bi = ["acquired", "subsidiary of"]
    confidence_bi, _ = run_propagation(
        g1_bi, g2_bi, embed_relation, name_embs, relations_bi
    )

    # Bidirectional should produce at least as much confidence
    mt_uni = ("g1u:Meridian Technologies", "g2u:Meridian Tech")
    mt_bi = ("g1b:Meridian Technologies", "g2b:Meridian Tech")
    assert confidence_bi[mt_bi] >= confidence_uni[mt_uni], (
        f"Bidirectional ({confidence_bi[mt_bi]}) should be >= "
        f"unidirectional ({confidence_uni[mt_uni]})"
    )


# ---------------------------------------------------------------------------
# Structural override of name dissimilarity
# ---------------------------------------------------------------------------


def test_shared_anchor_does_not_override_name_dissimilarity(embed, embed_relation):
    """A shared high-confidence anchor should not cause entities with
    different names to match.

    g1: "Dr. Priya Sharma"  --"founded"--> "NovaTech Labs"
    g2: "Dr. Elena Vasquez" --"founded"--> "NovaTech Labs"

    NovaTech Labs matches itself (identical names, confidence = 1.0).
    "founded" passes the relation gate. Structural evidence flows from
    the NovaTech anchor to the (Sharma, Vasquez) pair. Via noisy-OR,
    this overrides their name dissimilarity.

    Background graphs establish "founded" as a functional relation
    (mostly 1:1), matching the real pipeline where functionality is
    computed from all 53 graphs. Without the background, inverse
    functionality drops to 0.5 (two founders for NovaTech) and
    happens to prevent the match — but that's an artifact of the
    minimal setup, not of the algorithm being correct.

    Replicates the Dr. Sharma / Dr. Vasquez spurious merge.
    """
    g1 = make_graph("g1", [("Dr. Priya Sharma", "NovaTech Labs", "founded")])
    g2 = make_graph("g2", [("Dr. Elena Vasquez", "NovaTech Labs", "founded")])

    # Background: establish "founded" as a functional (1:1) relation
    bg_graphs = [
        make_graph(f"bg{i}", [(person, org, "founded")])
        for i, (person, org) in enumerate([
            ("Marcus Webb", "Alpha Corp"),
            ("Sarah Chen", "Beta Inc"),
            ("James Xu", "Gamma LLC"),
        ])
    ]

    name_embs = embed(
        ["Dr. Priya Sharma", "Dr. Elena Vasquez", "NovaTech Labs"]
    )

    rel_embs = {"founded": embed_relation("founded")}
    func = compute_functionality([g1, g2] + bg_graphs, rel_embs)

    # Premise: name similarity alone is below threshold
    sv_name_sim = float(np.dot(name_embs["Dr. Priya Sharma"], name_embs["Dr. Elena Vasquez"]))
    assert sv_name_sim < 0.8, (
        f"Premise failed: Sharma/Vasquez name_sim ({sv_name_sim}) >= 0.8"
    )

    confidence = propagate(g1, g2, name_embs, rel_embs, func)

    matches = select_matches(
        confidence, list(g1.entities), list(g2.entities), threshold=0.8
    )
    matched_names = {(g1.entities[a].name, g2.entities[b].name) for a, b in matches}

    # NovaTech Labs should NOT match — identical names alone are insufficient
    assert ("NovaTech Labs", "NovaTech Labs") not in matched_names, (
        "Identical names with no structural corroboration were incorrectly matched"
    )

    # Sharma and Vasquez should NOT match (different people)
    assert ("Dr. Priya Sharma", "Dr. Elena Vasquez") not in matched_names, (
        "Structural evidence from shared anchor overrode name dissimilarity: "
        "Dr. Priya Sharma matched Dr. Elena Vasquez"
    )


def test_similar_names_disjoint_neighborhoods_no_match(embed, embed_relation):
    """Near-identical names with zero structural overlap should not match.

    g1: "Dr. Elena Vasquez" --"is CEO of"--> "Volta Systems"
    g2: "Dr. Lena Vasquez"  --"is CEO of"--> "Halcyon Genomics"

    The names embed almost identically, but the neighbors share no
    similarity. Without structural corroboration, name similarity alone
    should not be sufficient.

    Replicates the Elena/Lena Vasquez false merge from real data: a
    quantum computing researcher merged with a biotech founder.
    """
    g1 = make_graph("g1", [("Dr. Elena Vasquez", "Volta Systems", "is CEO of")])
    g2 = make_graph("g2", [("Dr. Lena Vasquez", "Halcyon Genomics", "is CEO of")])

    name_embs = embed(
        ["Dr. Elena Vasquez", "Dr. Lena Vasquez", "Volta Systems", "Halcyon Genomics"],
    )

    # Premise: names are similar enough to trigger a match on their own
    name_sim = float(
        np.dot(name_embs["Dr. Elena Vasquez"], name_embs["Dr. Lena Vasquez"])
    )
    assert name_sim >= 0.8, (
        f"Premise failed: Elena/Lena name_sim ({name_sim:.3f}) below threshold — "
        f"the false merge must happen via a different path"
    )

    # Premise: neighbor names have no similarity
    nbr_sim = float(
        np.dot(name_embs["Volta Systems"], name_embs["Halcyon Genomics"])
    )
    assert nbr_sim < 0.5, (
        f"Premise failed: neighbor name_sim ({nbr_sim:.3f}) too high"
    )

    confidence, seeds = run_propagation(
        g1, g2, embed_relation, name_embs, ["is CEO of"]
    )

    matches = select_matches(
        confidence, list(g1.entities), list(g2.entities), threshold=0.8
    )
    matched_names = {(g1.entities[a].name, g2.entities[b].name) for a, b in matches}

    assert ("Dr. Elena Vasquez", "Dr. Lena Vasquez") not in matched_names, (
        f"Similar names with disjoint neighborhoods were incorrectly matched "
        f"(name_sim={name_sim:.3f}, no structural support)"
    )
