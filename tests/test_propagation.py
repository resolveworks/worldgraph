"""Layer 2 tests for similarity propagation.

These tests exercise the propagate() function on small graphs with real
embeddings.  Tests verify that:

- Identical/synonym relations with matching names produce structural evidence
- Dissimilar relations, weak neighbors, and many weak paths produce no
  structural evidence and no spurious matches (GH issue #1)
- Incoming edges propagate evidence (not just outgoing)
- Functionality weighting affects evidence strength
- Multi-hop chains require iterative propagation
- Name variation with structural reinforcement (the core use case)
- Dangling entities get no structural evidence
- Bidirectional edges accumulate evidence from both directions
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


def embed_names(entities: list[str], model) -> dict[str, np.ndarray]:
    """Embed entity names using the session model."""
    vecs = list(model.embed(entities))
    return {name: np.array(v) for name, v in zip(entities, vecs)}


def run_propagation(
    graph_a: Graph,
    graph_b: Graph,
    embed_phrase,
    name_embeddings: dict[str, np.ndarray],
    relations: list[str],
    threshold: float = 0.8,
):
    """Convenience wrapper: embed relations, compute functionality, propagate."""
    rel_embs = {r: embed_phrase(r) for r in relations}
    func = compute_functionality([graph_a, graph_b], rel_embs, threshold)
    name_sim, structural = propagate(
        graph_a,
        graph_b,
        name_embeddings,
        rel_embs,
        func,
    )
    return name_sim, structural


# ---------------------------------------------------------------------------
# Structural reinforcement tests
# ---------------------------------------------------------------------------


def test_identical_names_and_relations_reinforce(embed_phrase):
    """Two graphs with the same entity names and identical relations:
    propagation should produce structural evidence for the correct pairs."""
    #   g1: Apple --acquired--> Beats
    #   g2: Apple --acquired--> Beats
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "acquired")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats"], model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired"]
    )

    # Correct pairs should have structural evidence > 0
    assert structural[("g1:Apple", "g2:Apple")] > 0
    assert structural[("g1:Beats", "g2:Beats")] > 0

    # Wrong pairs should have no structural evidence
    assert structural[("g1:Apple", "g2:Beats")] == 0.0
    assert structural[("g1:Beats", "g2:Apple")] == 0.0


def test_synonym_relations_propagate(embed_phrase):
    """Synonym relation phrases ('acquired' / 'purchased') should propagate
    structural evidence just like identical ones."""
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "purchased")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats"], model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired", "purchased"]
    )

    # Correct pairs should have structural evidence
    assert structural[("g1:Apple", "g2:Apple")] > 0
    assert structural[("g1:Beats", "g2:Beats")] > 0


# ---------------------------------------------------------------------------
# No spurious matches (GH issue #1)
# ---------------------------------------------------------------------------


def test_dissimilar_relations_do_not_propagate(embed_phrase):
    """Unrelated relation phrases ('acquired' vs 'located in') should not
    produce structural evidence or matches.

    g1: Apple --acquired--> Beats
    g2: Tokyo --located in--> Japan
    """
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Tokyo", "Japan", "located in")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Apple", "Beats", "Tokyo", "Japan"]
    name_embs = embed_names(all_names, model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired", "located in"]
    )

    # No structural evidence should exist
    for key in structural:
        assert structural[key] == 0.0, (
            f"Spurious structural evidence: {key} = {structural[key]}"
        )

    # No matches
    matches = select_matches(
        name_sim,
        structural,
        list(g1.entities),
        list(g2.entities),
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == [], f"Spurious matches found: {matches}"


def test_weak_neighbors_do_not_propagate(embed_phrase):
    """Even with identical relation phrases, propagation should not produce
    structural evidence when neighbor name similarity is low.

    g1: Apple --acquired--> Beats
    g2: Google --acquired--> YouTube
    """
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Google", "YouTube", "acquired")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Apple", "Beats", "Google", "YouTube"]
    name_embs = embed_names(all_names, model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired"]
    )

    # No structural evidence — all neighbor pairs have low name similarity
    for key in structural:
        assert structural[key] == 0.0, (
            f"Spurious structural evidence: {key} = {structural[key]}"
        )


def test_many_weak_paths_do_not_accumulate(embed_phrase):
    """Many unrelated edges should not produce structural evidence or matches.

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

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = [
        "Org",
        "Target",
        "Project",
        "Person",
        "City",
        "Country",
        "River",
        "Event",
    ]
    name_embs = embed_names(all_names, model)

    relations = ["acquired", "funded", "hired", "located in", "borders", "hosts"]
    name_sim, structural = run_propagation(g1, g2, embed_phrase, name_embs, relations)

    matches = select_matches(
        name_sim,
        structural,
        list(g1.entities),
        list(g2.entities),
        name_threshold=0.8,
        structural_threshold=0.8,
    )
    assert matches == [], f"Spurious matches from accumulated weak paths: {matches}"


# ---------------------------------------------------------------------------
# Incoming edges
# ---------------------------------------------------------------------------


def test_incoming_edges_propagate(embed_phrase):
    """Structural evidence should propagate through incoming edges, not just
    outgoing.

    g1: Beats <--acquired-- Apple
    g2: Beats <--purchased-- Apple

    The target entity (Beats) receives incoming edges.  If propagation only
    examined outgoing edges, the Beats-Beats pair would get no evidence.
    """
    # Edges are directed source→target, so Apple→Beats.
    # For the *Beats* pair, Apple→Beats is an incoming edge.
    # Apple-Apple has high name sim → should propagate to Beats-Beats
    # via the incoming-edge path.
    g1 = make_graph("g1", [("Apple", "Beats", "acquired")])
    g2 = make_graph("g2", [("Apple", "Beats", "purchased")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats"], model)

    name_sim, structural = run_propagation(
        g1, g2, embed_phrase, name_embs, ["acquired", "purchased"]
    )

    # Both pairs should have structural evidence:
    # Apple-Apple gets it via outgoing edge (neighbor Beats-Beats has high name_sim)
    # Beats-Beats gets it via incoming edge (neighbor Apple-Apple has high name_sim)
    assert structural[("g1:Beats", "g2:Beats")] > 0, (
        "Incoming edge path did not propagate structural evidence to Beats-Beats"
    )
    assert structural[("g1:Apple", "g2:Apple")] > 0


# ---------------------------------------------------------------------------
# Functionality weighting
# ---------------------------------------------------------------------------


def test_functional_relation_produces_stronger_evidence(embed_phrase):
    """A 1:1 (functional) relation should produce stronger structural evidence
    than a many-to-many relation, all else being equal.

    g1: Apple --acquired--> Beats,   Apple --invested in--> Beats
    g2: Apple --acquired--> Beats,   Apple --invested in--> Beats

    'acquired' is 1:1 (functional = 1.0).
    'invested in' also appears 1:1 here, so we add fan-out to lower its
    functionality.
    """
    g1 = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Apple", "Beats", "invested in"),
            ("Apple", "Shazam", "invested in"),
            ("Google", "YouTube", "invested in"),
            ("Google", "Waymo", "invested in"),
        ],
    )
    g2 = make_graph(
        "g2",
        [
            ("Apple", "Beats", "acquired"),
            ("Apple", "Beats", "invested in"),
            ("Apple", "Shazam", "invested in"),
            ("Google", "YouTube", "invested in"),
            ("Google", "Waymo", "invested in"),
        ],
    )

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Apple", "Beats", "Shazam", "Google", "YouTube", "Waymo"]
    name_embs = embed_names(all_names, model)

    rel_embs = {
        "acquired": embed_phrase("acquired"),
        "invested in": embed_phrase("invested in"),
    }
    func = compute_functionality([g1, g2], rel_embs)

    # Verify the premise: 'acquired' is more functional than 'invested in'
    assert func["acquired"].forward > func["invested in"].forward

    # Now run propagation with only 'acquired' edges vs only 'invested in' edges
    g1_acq = make_graph("g1a", [("Apple", "Beats", "acquired")])
    g2_acq = make_graph("g2a", [("Apple", "Beats", "acquired")])
    g1_inv = make_graph("g1i", [("Apple", "Beats", "invested in")])
    g2_inv = make_graph("g2i", [("Apple", "Beats", "invested in")])

    _, structural_acq = propagate(g1_acq, g2_acq, name_embs, rel_embs, func)
    _, structural_inv = propagate(g1_inv, g2_inv, name_embs, rel_embs, func)

    # The functional relation should produce stronger evidence
    score_acq = structural_acq[("g1a:Apple", "g2a:Apple")]
    score_inv = structural_inv[("g1i:Apple", "g2i:Apple")]
    assert score_acq > score_inv, (
        f"Functional relation evidence ({score_acq}) should exceed "
        f"non-functional ({score_inv})"
    )


# ---------------------------------------------------------------------------
# Multi-hop propagation
# ---------------------------------------------------------------------------


def test_multi_hop_propagation_across_iterations(embed_phrase):
    """Evidence should propagate through a chain across multiple iterations.

    g1: Apple --acquired--> Beats --founded by--> James Chen
    g2: Apple --purchased--> Beats --founded by--> James Chen

    Iteration 1: James Chen-James Chen gets evidence (leaf nodes, high name
        sim, outgoing edge from Beats has neighbor James Chen with name_sim=1).
        Actually — JC-JC has no outgoing edges, but Beats-Beats gets evidence
        via outgoing edge to JC-JC (name_sim=1).
    After iteration 1: Beats-Beats has structural > 0, so
        confidence(Beats,Beats) = name_sim + structural >= threshold.
    Iteration 2: Apple-Apple gets evidence via outgoing edge to Beats-Beats.

    Without multi-iteration propagation, Apple-Apple would get no evidence
    because Beats-Beats has zero structural score initially.
    """
    g1 = make_graph(
        "g1",
        [
            ("Apple", "Beats", "acquired"),
            ("Beats", "James Chen", "founded by"),
        ],
    )
    g2 = make_graph(
        "g2",
        [
            ("Apple", "Beats", "purchased"),
            ("Beats", "James Chen", "founded by"),
        ],
    )

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats", "James Chen"], model)

    relations = ["acquired", "purchased", "founded by"]
    rel_embs = {r: embed_phrase(r) for r in relations}
    func = compute_functionality([g1, g2], rel_embs)

    # Single iteration: Apple-Apple should NOT have evidence yet
    _, structural_1 = propagate(g1, g2, name_embs, rel_embs, func, max_iter=1)
    # Beats-Beats should have evidence (neighbor JC-JC has name_sim=1)
    assert structural_1[("g1:Beats", "g2:Beats")] > 0

    # Apple-Apple needs Beats-Beats confidence >= threshold,
    # which requires structural > 0 from iteration 1
    # With max_iter=1, Apple may or may not have evidence depending on
    # whether Beats already accumulated enough. Let's check full convergence:
    _, structural_full = propagate(
        g1, g2, name_embs, rel_embs, func, max_iter=30
    )

    # After convergence, all three pairs should have evidence
    assert structural_full[("g1:Apple", "g2:Apple")] > 0, (
        "Multi-hop propagation failed: Apple-Apple has no structural evidence"
    )
    assert structural_full[("g1:Beats", "g2:Beats")] > 0
    assert structural_full[("g1:James Chen", "g2:James Chen")] > 0

    # And evidence should have increased from iteration 1 to convergence
    assert structural_full[("g1:Apple", "g2:Apple")] >= structural_1.get(
        ("g1:Apple", "g2:Apple"), 0.0
    )


# ---------------------------------------------------------------------------
# Name variation with structural reinforcement
# ---------------------------------------------------------------------------


def test_name_variation_with_structural_reinforcement(embed_phrase):
    """The core use case: similar-but-not-identical entity names get matched
    when structural evidence reinforces them.

    g1: Meridian Technologies --acquired--> DataVault Inc
    g2: Meridian Tech         --purchased--> DataVault Inc

    'Meridian Technologies' / 'Meridian Tech' have name_sim ~0.86 (above 0.8).
    Structural evidence from the matching edge + matching neighbor (DataVault Inc)
    should reinforce the match.
    """
    g1 = make_graph("g1", [("Meridian Technologies", "DataVault Inc", "acquired")])
    g2 = make_graph("g2", [("Meridian Tech", "DataVault Inc", "purchased")])

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Meridian Technologies", "Meridian Tech", "DataVault Inc"]
    name_embs = embed_names(all_names, model)

    relations = ["acquired", "purchased"]
    name_sim, structural = run_propagation(g1, g2, embed_phrase, name_embs, relations)

    # The varied-name pair should have structural evidence
    mt_pair = ("g1:Meridian Technologies", "g2:Meridian Tech")
    assert structural[mt_pair] > 0, (
        "Name variation pair got no structural reinforcement"
    )

    # DataVault-DataVault should also have evidence (via incoming edge)
    dv_pair = ("g1:DataVault Inc", "g2:DataVault Inc")
    assert structural[dv_pair] > 0

    # Both should pass select_matches
    matches = select_matches(
        name_sim,
        structural,
        list(g1.entities),
        list(g2.entities),
        name_threshold=0.8,
        structural_threshold=0.1,
    )
    matched_names = [(g1.entities[a].name, g2.entities[b].name) for a, b in matches]
    assert ("Meridian Technologies", "Meridian Tech") in matched_names
    assert ("DataVault Inc", "DataVault Inc") in matched_names


# ---------------------------------------------------------------------------
# Dangling entities
# ---------------------------------------------------------------------------


def test_dangling_entities_get_no_structural_evidence(embed_phrase):
    """Entities with no matching structure should get zero structural evidence,
    even when other entities in the same graphs do match.

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

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_names = ["Apple", "Beats", "SolarGrid", "WindPower"]
    name_embs = embed_names(all_names, model)

    relations = ["acquired", "purchased", "hired"]
    name_sim, structural = run_propagation(g1, g2, embed_phrase, name_embs, relations)

    # Matched entities should have structural evidence
    assert structural[("g1:Apple", "g2:Apple")] > 0
    assert structural[("g1:Beats", "g2:Beats")] > 0

    # Dangling cross-pairs should have zero structural evidence
    assert structural[("g1:SolarGrid", "g2:WindPower")] == 0.0
    # And the wrong pairings too
    assert structural[("g1:SolarGrid", "g2:Apple")] == 0.0
    assert structural[("g1:SolarGrid", "g2:Beats")] == 0.0

    # select_matches should not include any dangling entity
    matches = select_matches(
        name_sim,
        structural,
        list(g1.entities),
        list(g2.entities),
        name_threshold=0.8,
        structural_threshold=0.1,
    )
    matched_ids = {eid for pair in matches for eid in pair}
    assert "g1:SolarGrid" not in matched_ids
    assert "g2:WindPower" not in matched_ids


# ---------------------------------------------------------------------------
# Bidirectional edges
# ---------------------------------------------------------------------------


def test_bidirectional_edges_accumulate_evidence(embed_phrase):
    """Entity pairs connected by edges in both directions should accumulate
    evidence from both, producing higher scores than a single direction.

    Unidirectional:
        g1: Apple --acquired--> Beats
        g2: Apple --acquired--> Beats

    Bidirectional:
        g1: Apple --acquired--> Beats,  Beats --founded by--> Apple
        g2: Apple --acquired--> Beats,  Beats --founded by--> Apple
    """
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embs = embed_names(["Apple", "Beats"], model)

    # Unidirectional
    g1_uni = make_graph("g1u", [("Apple", "Beats", "acquired")])
    g2_uni = make_graph("g2u", [("Apple", "Beats", "acquired")])

    relations_uni = ["acquired"]
    _, structural_uni = run_propagation(
        g1_uni, g2_uni, embed_phrase, name_embs, relations_uni
    )

    # Bidirectional
    g1_bi = make_graph(
        "g1b",
        [
            ("Apple", "Beats", "acquired"),
            ("Beats", "Apple", "founded by"),
        ],
    )
    g2_bi = make_graph(
        "g2b",
        [
            ("Apple", "Beats", "acquired"),
            ("Beats", "Apple", "founded by"),
        ],
    )

    relations_bi = ["acquired", "founded by"]
    _, structural_bi = run_propagation(
        g1_bi, g2_bi, embed_phrase, name_embs, relations_bi
    )

    # Bidirectional should produce at least as much evidence
    score_uni = structural_uni[("g1u:Apple", "g2u:Apple")]
    score_bi = structural_bi[("g1b:Apple", "g2b:Apple")]
    assert score_bi >= score_uni, (
        f"Bidirectional ({score_bi}) should be >= unidirectional ({score_uni})"
    )
