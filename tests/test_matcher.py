"""Matcher tests: damped pairwise propagation over the recursive term model.

Pairs are cross-graph and same-kind (entity↔entity, statement↔statement).
Entity pairs seed from name similarity, statement pairs from an
injectable predicate prior (neutral when none is given). The position a
term occupies in a statement is part of its evidence signature: slots
align only with the same slot, so direction contradictions are negative
evidence, and nesting (qualifiers, denials) propagates through the same
mechanism as any participant. Similarity only proposes — merging
requires structurally tested neighbors, and union-find inside
propagation is the sole merge authority.
"""

import pytest

from worldgraph.constants import NEUTRAL_PRIOR
from worldgraph.graph import Graph, Statement, Term
from worldgraph.match import match_graphs

Qid = tuple[str, str]


def key(a: Term, b: Term) -> tuple[Qid, Qid]:
    """Confidence-dict key for a pair of terms."""
    return (a.graph_id, a.id), (b.graph_id, b.id)


def same_group(groups: list[list[Qid]], *terms: Term) -> bool:
    """True iff all given terms ended up in one match group."""
    wanted = {(t.graph_id, t.id) for t in terms}
    return any(wanted <= set(group) for group in groups)


# ---------------------------------------------------------------------------
# Pinned spec: similarity proposes, structure disposes
# ---------------------------------------------------------------------------


def test_prior_alone_never_merges():
    """A 0.95 predicate prior proposes two statement pairs, but the
    participants do not correspond: no cross-article name seeds, so every
    slot-aligned counterpart search fails — negative evidence. Nothing
    merges."""
    g1 = Graph(id="g1")
    stellar1 = g1.add_entity("Stellar Foods")
    nova1 = g1.add_entity("Nova Markets")
    g1.add_statement(stellar1, "acquire", nova1)
    orbit1 = g1.add_entity("Orbit Logistics")
    port1 = g1.add_entity("Eastport")
    g1.add_statement(orbit1, "lease", port1)

    g2 = Graph(id="g2")
    canyon2 = g2.add_entity("Canyon Media")
    pulse2 = g2.add_entity("Pulse Radio")
    g2.add_statement(canyon2, "acquire", pulse2)
    quay2 = g2.add_entity("Quay Partners")
    dock2 = g2.add_entity("Northdock")
    g2.add_statement(quay2, "lease", dock2)

    def hot_prior(a: Statement, b: Statement) -> float:
        return 0.95

    _confidence, groups, _ = match_graphs([g1, g2], predicate_prior=hot_prior)

    assert groups == []


def test_direction_contradiction_never_merges():
    """(council, 'govern', district) vs (district, 'govern', council):
    identical names, identical predicate, opposite direction. Each slot's
    counterpart search finds the wrong entity (negative evidence), and
    the name-identical entities find no slot-matched counterpart at all —
    council is subject in one graph, object in the other — so nothing
    merges, entities included."""
    g1 = Graph(id="g1")
    council1 = g1.add_entity("City Council")
    district1 = g1.add_entity("Harbor District")
    govern1 = g1.add_statement(council1, "govern", district1)

    g2 = Graph(id="g2")
    council2 = g2.add_entity("City Council")
    district2 = g2.add_entity("Harbor District")
    govern2 = g2.add_statement(district2, "govern", council2)

    confidence, groups, _ = match_graphs([g1, g2])

    assert not same_group(groups, govern1, govern2)
    assert not same_group(groups, council1, council2)
    assert not same_group(groups, district1, district2)
    assert confidence[key(govern1, govern2)] < NEUTRAL_PRIOR


def test_paraphrase_merges_under_neutral_prior():
    """(acme, 'acquire', beta) vs (acme, 'purchase', beta): predicates are
    never compared — the slot-aligned identical participants alone carry
    the pair over the merge bar."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    beta1 = g1.add_entity("Beta Systems")
    acquire1 = g1.add_statement(acme1, "acquire", beta1)

    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp")
    beta2 = g2.add_entity("Beta Systems")
    acquire2 = g2.add_statement(acme2, "purchase", beta2)

    confidence, groups, _ = match_graphs([g1, g2])

    assert same_group(groups, acquire1, acquire2)
    assert same_group(groups, acme1, acme2)
    assert same_group(groups, beta1, beta2)
    assert confidence[key(acquire1, acquire2)] > NEUTRAL_PRIOR


def test_paraphrase_merges_with_high_prior():
    """Same paraphrase pair, seeded by a 0.9 predicate prior: the prior
    spends its headroom, but the merge is still gated on the tested
    slot-aligned participants."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    beta1 = g1.add_entity("Beta Systems")
    acquire1 = g1.add_statement(acme1, "acquire", beta1)

    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp")
    beta2 = g2.add_entity("Beta Systems")
    acquire2 = g2.add_statement(acme2, "purchase", beta2)

    def acquire_purchase_prior(a: Statement, b: Statement) -> float:
        if {a.predicate, b.predicate} == {"acquire", "purchase"}:
            return 0.9
        return NEUTRAL_PRIOR

    _confidence, groups, _ = match_graphs([g1, g2], predicate_prior=acquire_purchase_prior)

    assert same_group(groups, acquire1, acquire2)


# ---------------------------------------------------------------------------
# Pinned spec: disagreement and its positive mirror
# ---------------------------------------------------------------------------


def test_qualifier_disagreement_keeps_facts_apart():
    """(acme, 'be headquartered in', Berlin) vs (acme, 'be headquartered
    in', Munich): the shared subject is outweighed by the mismatched
    object — negative evidence is decisive — so the HQ statements stay
    apart, and Berlin never merges with Munich."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    berlin1 = g1.add_entity("Berlin")
    hq1 = g1.add_statement(acme1, "be headquartered in", berlin1)

    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp")
    munich2 = g2.add_entity("Munich")
    hq2 = g2.add_statement(acme2, "be headquartered in", munich2)

    confidence, groups, _ = match_graphs([g1, g2])

    assert not same_group(groups, hq1, hq2)
    assert not same_group(groups, berlin1, munich2)
    assert confidence[key(hq1, hq2)] < NEUTRAL_PRIOR


def test_shared_rare_participant_corroborates():
    """Positive mirror of the HQ disagreement: two reports of the same
    deal share the slot-aligned price qualifier — a rare periphery
    participant — which corroborates the deal pair through its subject
    slot, and the qualifier pair lifts in turn through the merged deal."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    beta1 = g1.add_entity("Beta Systems")
    deal1 = g1.add_statement(acme1, "acquire", beta1)
    price1 = g1.add_entity("$4.2 billion")
    price_stmt1 = g1.add_statement(deal1, "for", price1)

    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp")
    beta2 = g2.add_entity("Beta Systems")
    deal2 = g2.add_statement(acme2, "purchase", beta2)
    price2 = g2.add_entity("$4.2 billion")
    price_stmt2 = g2.add_statement(deal2, "for", price2)

    _confidence, groups, _ = match_graphs([g1, g2])

    assert same_group(groups, deal1, deal2)
    assert same_group(groups, price_stmt1, price_stmt2)
    assert same_group(groups, price1, price2)


# ---------------------------------------------------------------------------
# Pinned spec: recursion is free
# ---------------------------------------------------------------------------
# Pinned spec: axioms
# ---------------------------------------------------------------------------


def test_identical_names_alone_never_merge():
    """Identical entity names with zero corroborating statements: no
    structural evidence, no merge — regardless of the 1.0 name seed."""
    g1 = Graph(id="g1")
    g1.add_entity("Acme Corp")
    g2 = Graph(id="g2")
    g2.add_entity("Acme Corp")

    _confidence, groups, _ = match_graphs([g1, g2])

    assert groups == []


# ---------------------------------------------------------------------------
# No spurious matches
# ---------------------------------------------------------------------------


def test_unrelated_graphs_do_not_match():
    """Entirely disjoint terms: all counterpart searches fail, nothing
    merges."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    beta1 = g1.add_entity("Beta Systems")
    g1.add_statement(acme1, "acquire", beta1)

    g2 = Graph(id="g2")
    tokyo2 = g2.add_entity("Tokyo")
    japan2 = g2.add_entity("Japan")
    g2.add_statement(tokyo2, "be located in", japan2)

    _confidence, groups, _ = match_graphs([g1, g2])

    assert groups == []


def test_mismatched_participant_suppresses_statement_and_shared_name():
    """'John purchased a car' vs 'John purchased a truck': the matched
    subject path is outweighed by the mismatched object path — the
    statement pair drops below the neutral prior, and the same-name Johns
    (whose only structural interaction is the suppressed statement) do
    not merge either."""
    g1 = Graph(id="g1")
    john1 = g1.add_entity("John Mercer")
    car1 = g1.add_entity("car")
    purchase1 = g1.add_statement(john1, "purchase", car1)

    g2 = Graph(id="g2")
    john2 = g2.add_entity("John Mercer")
    truck2 = g2.add_entity("truck")
    purchase2 = g2.add_statement(john2, "purchase", truck2)

    confidence, groups, _ = match_graphs([g1, g2])

    assert confidence[key(purchase1, purchase2)] < NEUTRAL_PRIOR
    assert not same_group(groups, purchase1, purchase2)
    assert not same_group(groups, john1, john2)


# ---------------------------------------------------------------------------
# Multi-source and calibration
# ---------------------------------------------------------------------------


def test_three_sources_merge_transitively():
    """Three outlets reporting the same deal with three predicates: all
    pairwise lifts cross the bar, and union-find commits one group of
    three statements."""
    graphs = []
    statements = []
    for i, predicate in enumerate(["acquire", "purchase", "take over"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        beta = g.add_entity("Beta Systems")
        statements.append(g.add_statement(acme, predicate, beta))
        graphs.append(g)

    _confidence, groups, _ = match_graphs(graphs)

    assert same_group(groups, *statements)


def test_hub_participants_weaken_slot_evidence():
    """A participant appearing in many statements is a weak identity
    signal. The same acquire/purchase pair merges in a two-graph corpus,
    but not when both participants are hubs — each the subject and object
    of six further statements in background graphs."""
    def deal_graphs(run: str) -> tuple[list[Graph], Statement, Statement]:
        g1 = Graph(id=f"{run}-a1")
        acme1 = g1.add_entity("Acme Corp")
        gamma1 = g1.add_entity("Gamma AI")
        acquire1 = g1.add_statement(acme1, "acquire", gamma1)
        g2 = Graph(id=f"{run}-a2")
        acme2 = g2.add_entity("Acme Corp")
        gamma2 = g2.add_entity("Gamma AI")
        acquire2 = g2.add_statement(acme2, "purchase", gamma2)
        return [g1, g2], acquire1, acquire2

    clean, acquire1, acquire2 = deal_graphs("clean")
    conf_clean, groups_clean, _ = match_graphs(clean)
    assert same_group(groups_clean, acquire1, acquire2)

    graphs, acquire3, acquire4 = deal_graphs("hub")
    for i, predicate in enumerate(
        [
            "sign agreement with",
            "partner with",
            "jointly develop",
            "co-market",
            "form alliance with",
            "share premises with",
        ]
    ):
        bg = Graph(id=f"hub-bg{i}")
        acme = bg.add_entity("Acme Corp")
        gamma = bg.add_entity("Gamma AI")
        bg.add_statement(acme, predicate, gamma)
        graphs.append(bg)

    conf_hub, groups_hub, _ = match_graphs(graphs)

    assert not same_group(groups_hub, acquire3, acquire4)
    assert conf_hub[key(acquire3, acquire4)] < conf_clean[key(acquire1, acquire2)]


# ---------------------------------------------------------------------------
# Invariants and degenerate inputs
# ---------------------------------------------------------------------------


def _scenario() -> list[Graph]:
    """Three graphs: two reports of the same deal (which merge) and one
    unrelated deal (which does not)."""
    graphs = []
    for i, (buyer, target, predicate) in enumerate(
        [
            ("Meridian Corp", "DataVault", "acquire"),
            ("Meridian Corp", "DataVault", "purchase"),
            ("NexGen Holdings", "ClearSky", "acquire"),
        ]
    ):
        g = Graph(id=f"g{i}")
        buyer_term = g.add_entity(buyer)
        target_term = g.add_entity(target)
        g.add_statement(buyer_term, predicate, target_term)
        graphs.append(g)
    return graphs


def test_confidence_is_symmetric():
    """For every pair key, the reversed key exists with the same score —
    consumers index confidence with arbitrary pair orderings."""
    confidence, _, _ = match_graphs(_scenario())

    assert confidence
    for (a, b), score in confidence.items():
        assert (b, a) in confidence
        assert confidence[(b, a)] == score


def test_graph_order_does_not_change_results():
    """Permutation invariance: the input order of graphs cannot change
    groups or confidences."""
    import itertools

    graphs = _scenario()
    conf_ref, groups_ref, _ = match_graphs(graphs)
    ref = {frozenset(g) for g in groups_ref}

    for permutation in itertools.permutations(graphs):
        conf, groups, _ = match_graphs(list(permutation))
        assert {frozenset(g) for g in groups} == ref
        assert conf == conf_ref


def test_identical_runs_are_deterministic():
    graphs = _scenario()
    conf_a, groups_a, merged_a = match_graphs(graphs)
    conf_b, groups_b, merged_b = match_graphs(graphs)

    assert conf_a == conf_b
    assert groups_a == groups_b
    assert merged_a.terms.keys() == merged_b.terms.keys()


def test_single_graph_matches_nothing():
    """No cross-graph pairs exist for a single graph."""
    g = Graph(id="g1")
    acme = g.add_entity("Acme Corp")
    g.add_statement(acme, "acquire", g.add_entity("Beta Systems"))

    confidence, groups, merged = match_graphs([g])

    assert confidence == {}
    assert groups == []
    assert len(merged.terms) == 3


def test_no_graphs_is_empty():
    confidence, groups, merged = match_graphs([])

    assert confidence == {}
    assert groups == []
    assert merged.terms == {}


def test_duplicate_graph_ids_raise():
    """Qualified ids need unique graph ids — duplicate graph ids are
    invalid input (the graphs would silently never pair)."""
    g1 = Graph(id="g1")
    g1.add_entity("Acme Corp")
    g2 = Graph(id="g1")
    g2.add_entity("Beta Systems")

    with pytest.raises(ValueError, match="duplicate graph id"):
        match_graphs([g1, g2])
