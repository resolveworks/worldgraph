"""Matcher laws: the mechanism of structural matching, quantified over parameters.

Doctrine: tests prove the mechanism, evals (evals/matcher) determine
calibration. No test here assumes a value of ``MatcherParams`` — every
assertion takes one of three lawful forms:

- **Confidence laws**: orderings and relations (sub-neutral, strictly
  raises, strictly weakens) that must hold under any parameters, checked
  under CALIBRATION and deliberately different variants. Confidence is
  measured with merging frozen out (merge_threshold above 1), so the
  propagated score is never replaced by the merged-pair convention of
  reporting 1.0.
- **Gate laws**: outcomes forced by structure alone. merge_threshold is
  swept across a grid inside the test — the full [0, 1] range (0.0
  included) where the gate is purely structural, thresholds at or above
  the neutral prior where the gate is backed by negative evidence.
- **Can-merge laws**: proofs of capability under explicit extremes —
  loose parameters (threshold barely above the neutral prior, weak
  shrinkage) for what the machinery MUST be able to merge, an
  unreachable threshold for what it must not.

Whether a given corpus merges under the canonical calibration is a data
question owned by evals/matcher; scenario outcomes are not asserted here.

The mechanism under law: pairs are cross-graph and same-kind; entity
pairs seed from name similarity, statement pairs from an injectable
predicate prior (neutral when none is given). Slots align only with the
same slot, so direction contradictions are negative evidence and nesting
propagates through the same mechanism as any participant. Similarity
only proposes — merging requires confidence >= merge_threshold AND at
least one structurally tested neighbor, and union-find inside propagation
is the sole merge authority. Negative evidence is decisive at full
weight; positive evidence is shrunk by corroboration count.
"""

from typing import NamedTuple

import pytest

from worldgraph.constants import NEUTRAL_PRIOR
from worldgraph.graph import Entity, Graph, Statement, Term
from worldgraph.match import CALIBRATION, MatcherParams, match_graphs

Qid = tuple[str, str]


def key(a: Term, b: Term) -> tuple[Qid, Qid]:
    """Confidence-dict key for a pair of terms."""
    return (a.graph_id, a.id), (b.graph_id, b.id)


def same_group(groups: list[list[Qid]], *terms: Term) -> bool:
    """True iff all given terms ended up in one match group."""
    wanted = {(t.graph_id, t.id) for t in terms}
    return any(wanted <= set(group) for group in groups)


# Deliberately different, non-degenerate parameter sets. A relation
# asserted under all three quantifies over params: the canonical
# calibration, a fast-saturating weakly-shrunk set, and a slow-saturating
# strongly-shrunk one.
VARIANTS = [
    CALIBRATION,
    CALIBRATION._replace(prior_strength=0.25, exp_lambda=5.0, damping=0.3, max_iter=60),
    CALIBRATION._replace(prior_strength=4.0, exp_lambda=0.5, damping=0.8, max_iter=120),
]
VARIANT_IDS = ["calibration", "fast-saturation", "slow-saturation"]


def frozen(params: MatcherParams) -> MatcherParams:
    """Measurement instrument for confidence laws: the same propagation
    with merging made impossible, so propagated scores are never
    replaced by the merged-pair convention (merged pairs report 1.0)."""
    return params._replace(merge_threshold=1.01)


# The full [0, 1] range, 0.0 included: a pair with zero structurally
# tested neighbors never merges at ANY threshold — the structural gate
# (n_tested > 0) is ANDed with the score test, so even a 1.0 seed at
# threshold 0.0 stays split.
STRUCTURAL_GRID = [0.0, 0.1, 0.5, 0.9, 1.0]

# Thresholds at or above the neutral prior: the domain of "never merges"
# laws backed by negative evidence. Below the neutral prior a
# sub-neutral-confidence pair still satisfies conf >= threshold, and a
# pair with tested neighbors merges on net-negative evidence — the
# demonstrated boundary is documented in test_prior_alone_never_merges.
NEUTRAL_GRID = [NEUTRAL_PRIOR, 0.7, 0.9, 1.0]

# Explicit extreme for can-merge laws: the bar sits barely above the
# neutral prior and shrinkage is weak, so anything the mechanism can
# lift at all clears it — assertions are about capability, never about
# calibration.
LOOSE = MatcherParams(
    merge_threshold=NEUTRAL_PRIOR + 0.01,
    prior_strength=0.25,
    exp_lambda=2.0,
    damping=0.5,
    max_iter=50,
    epsilon=1e-6,
)


class DealCorpus(NamedTuple):
    """Two or more graphs reporting the same deal between identically
    named participants, differing only in predicate wording."""

    graphs: list[Graph]
    subjects: list[Entity]
    objects: list[Entity]
    deals: list[Statement]


def _deal_corpus(predicates: list[str]) -> DealCorpus:
    graphs, subjects, objects, deals = [], [], [], []
    for i, predicate in enumerate(predicates):
        g = Graph(id=f"g{i}")
        subject = g.add_entity("Acme Corp")
        object_ = g.add_entity("Beta Systems")
        deals.append(g.add_statement(subject, predicate, object_))
        graphs.append(g)
        subjects.append(subject)
        objects.append(object_)
    return DealCorpus(graphs, subjects, objects, deals)


def _hot_prior(a: Statement, b: Statement) -> float:
    """A 0.95 predicate prior proposing every statement pair equally."""
    return 0.95


def _acquire_purchase_prior(a: Statement, b: Statement) -> float:
    """A 0.9 prior proposing exactly the acquire/purchase paraphrase."""
    if {*a.predicates, *b.predicates} == {"acquire", "purchase"}:
        return 0.9
    return NEUTRAL_PRIOR


def _direction_graphs() -> tuple[
    list[Graph], tuple[Term, Term, Term], tuple[Term, Term, Term]
]:
    """(council, 'govern', district) vs the same fact reported
    backwards: identical names, identical predicate, opposite
    direction."""
    g1 = Graph(id="g1")
    council1 = g1.add_entity("City Council")
    district1 = g1.add_entity("Harbor District")
    govern1 = g1.add_statement(council1, "govern", district1)
    g2 = Graph(id="g2")
    council2 = g2.add_entity("City Council")
    district2 = g2.add_entity("Harbor District")
    govern2 = g2.add_statement(district2, "govern", council2)
    return [g1, g2], (council1, district1, govern1), (council2, district2, govern2)


# ---------------------------------------------------------------------------
# Gate laws: zero structurally tested neighbors (full threshold range)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_identical_names_alone_never_merge(params):
    """Lone same-name entities with zero statements never merge — at any
    threshold in the full grid, under any variant — and their confidence
    sits untouched at the name seed. With no statements there is no
    adjacency and no counterpart search: n_tested == 0, so the structural
    gate blocks even the maximal 1.0 seed at threshold 0.0. Any lower
    seed is blocked a fortiori."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp")

    for threshold in STRUCTURAL_GRID:
        confidence, groups, _ = match_graphs(
            [g1, g2], params=params._replace(merge_threshold=threshold)
        )
        assert groups == [], f"merged at threshold {threshold}"
        assert confidence[key(acme1, acme2)] == pytest.approx(1.0)


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_crossed_slots_are_never_tested(params):
    """Gate law: name-identical entities sitting in opposite slots of
    two reports of the same fact never merge, at any threshold in the
    full grid. Council is subject in one graph and object in the other,
    so its slot-aligned counterpart search finds nothing — n_tested == 0
    — while its confidence sits untouched at the 1.0 name seed: a score
    that qualifies at every threshold still cannot merge without a
    structurally tested neighbor."""
    graphs, (council1, _, _), (council2, _, _) = _direction_graphs()

    for threshold in STRUCTURAL_GRID:
        confidence, groups, _ = match_graphs(
            graphs, params=params._replace(merge_threshold=threshold)
        )
        assert not same_group(groups, council1, council2), (
            f"merged at threshold {threshold}"
        )
        assert confidence[key(council1, council2)] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate laws: negative evidence keeps pairs apart (thresholds at or above
# the neutral prior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_direction_contradiction_is_negative_evidence(params):
    """The two govern statements have slot-aligned counterparts on the
    wrong side of every slot (subject finds object, object finds
    subject) at name similarity 0: negative evidence drives the pair
    below the neutral prior under every variant, and it never merges for
    any threshold at or above the neutral prior.

    Domain discovery: at threshold 0.0 the pair merges — the gate is
    (confidence >= threshold AND n_tested > 0), so once the threshold
    drops below the pair's sub-neutral confidence, net-negative evidence
    qualifies. Negative evidence suppresses; it does not veto."""
    graphs, (_, _, govern1), (_, _, govern2) = _direction_graphs()

    confidence, _, _ = match_graphs(graphs, params=frozen(params))
    assert confidence[key(govern1, govern2)] < NEUTRAL_PRIOR

    for threshold in NEUTRAL_GRID:
        _, groups, _ = match_graphs(
            graphs, params=params._replace(merge_threshold=threshold)
        )
        assert not same_group(groups, govern1, govern2), (
            f"merged at threshold {threshold}"
        )


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_prior_alone_never_merges(params):
    """Gate law: a 0.95 predicate prior proposes the statement pair, but
    the participants cannot correspond (cross-name similarity 0), so
    every slot-aligned counterpart search contributes negative evidence:
    the prior is spent driving the pair below the neutral prior under
    every variant, and nothing merges — statements or entities — for any
    threshold at or above the neutral prior.

    Domain discovery: the law's domain stops at the neutral prior. Under
    weak saturation (exp_lambda = 0.5) the pair rests near 0.35: at
    threshold 0.1 it merges, prior and negative evidence notwithstanding
    — net-negative confidence satisfies conf >= threshold once the
    threshold drops below it."""
    g1 = Graph(id="g1")
    acquire1 = g1.add_statement(
        g1.add_entity("Stellar Foods"), "acquire", g1.add_entity("Nova Markets")
    )
    g2 = Graph(id="g2")
    acquire2 = g2.add_statement(
        g2.add_entity("Canyon Media"), "acquire", g2.add_entity("Pulse Radio")
    )

    confidence, _, _ = match_graphs(
        [g1, g2], params=frozen(params), predicate_prior=_hot_prior
    )
    assert confidence[key(acquire1, acquire2)] < NEUTRAL_PRIOR

    for threshold in NEUTRAL_GRID:
        _, groups, _ = match_graphs(
            [g1, g2],
            params=params._replace(merge_threshold=threshold),
            predicate_prior=_hot_prior,
        )
        assert groups == [], f"merged at threshold {threshold}"


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_unrelated_graphs_do_not_match(params):
    """Gate law: disjoint vocabularies — every slot-aligned counterpart
    search lands on a 0-similarity name, negative evidence all the way
    down. Nothing merges for any threshold at or above the neutral
    prior. Same domain discovery as the prior-alone law: below the
    neutral prior, net-negative pairs with tested neighbors merge."""
    g1 = Graph(id="g1")
    acquire1 = g1.add_statement(
        g1.add_entity("Acme Corp"), "acquire", g1.add_entity("Beta Systems")
    )
    g2 = Graph(id="g2")
    located2 = g2.add_statement(
        g2.add_entity("Tokyo"), "be located in", g2.add_entity("Japan")
    )

    confidence, _, _ = match_graphs([g1, g2], params=frozen(params))
    assert confidence[key(acquire1, located2)] < NEUTRAL_PRIOR

    for threshold in NEUTRAL_GRID:
        _, groups, _ = match_graphs(
            [g1, g2], params=params._replace(merge_threshold=threshold)
        )
        assert groups == [], f"merged at threshold {threshold}"


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_qualifier_disagreement_keeps_facts_apart(params):
    """(acme, 'be headquartered in', Berlin) vs (acme, ..., Munich): the
    shared subject pools by name across both reports, so it carries less
    inverse functionality than the unique mismatched object — the
    subject's positive evidence is outweighed by the object's negative
    evidence, and the HQ pair sits below the neutral prior under every
    variant. Neither the statements nor the disagreeing cities merge for
    any threshold at or above the neutral prior. The shared subject
    itself has no gate (see test_mismatched_participant_is_decisive)."""
    graphs, hqs, cities = [], [], []
    for i, city in enumerate(["Berlin", "Munich"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        city_term = g.add_entity(city)
        hqs.append(g.add_statement(acme, "be headquartered in", city_term))
        cities.append(city_term)
        graphs.append(g)

    confidence, _, _ = match_graphs(graphs, params=frozen(params))
    assert confidence[key(hqs[0], hqs[1])] < NEUTRAL_PRIOR

    for threshold in NEUTRAL_GRID:
        _, groups, _ = match_graphs(
            graphs, params=params._replace(merge_threshold=threshold)
        )
        assert not same_group(groups, hqs[0], hqs[1]), (
            f"HQ statements merged at threshold {threshold}"
        )
        assert not same_group(groups, cities[0], cities[1]), (
            f"cities merged at threshold {threshold}"
        )


# ---------------------------------------------------------------------------
# Confidence laws: orderings that hold under any parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_identical_participants_lift_paraphrase_above_neutral(params):
    """Confidence law: predicates are never compared by the propagation
    core — the slot-aligned identical participants alone carry the
    paraphrase pair (acquire/purchase) strictly above the neutral prior,
    under every variant, with no predicate prior injected."""
    corpus = _deal_corpus(["acquire", "purchase"])

    confidence, _, _ = match_graphs(corpus.graphs, params=frozen(params))

    assert confidence[key(corpus.deals[0], corpus.deals[1])] > NEUTRAL_PRIOR


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_corroborating_path_raises_confidence(params):
    """Confidence law: adding a corroborating path — the shared price
    qualifier (deal, 'for', '$4.2 billion') nesting through the deal's
    participant slots — strictly raises the deal pair's confidence over
    the same pair in the plain two-participant corpus, under every
    variant. Positive evidence aggregates per tested slot-aligned
    neighbor: a third corroborating path adds, never subtracts."""
    plain = _deal_corpus(["acquire", "purchase"])
    graphs, deals, _qualifiers, _prices = [], [], [], []
    for i, predicate in enumerate(["acquire", "purchase"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        beta = g.add_entity("Beta Systems")
        deal = g.add_statement(acme, predicate, beta)
        price = g.add_entity("$4.2 billion")
        g.add_statement(deal, "for", price)
        deals.append(deal)
        graphs.append(g)

    conf_plain, _, _ = match_graphs(plain.graphs, params=frozen(params))
    conf_with, _, _ = match_graphs(graphs, params=frozen(params))

    pair = key(deals[0], deals[1])
    assert conf_with[pair] > conf_plain[key(plain.deals[0], plain.deals[1])]


def _purchase_corpus(
    objects: list[str],
) -> tuple[list[Graph], list[Entity], list[Statement]]:
    """'John Mercer purchased <object>' reported in two graphs."""
    graphs, johns, purchases = [], [], []
    for i, object_ in enumerate(objects):
        g = Graph(id=f"g{i}")
        john = g.add_entity("John Mercer")
        purchases.append(g.add_statement(john, "purchase", g.add_entity(object_)))
        johns.append(john)
        graphs.append(g)
    return graphs, johns, purchases


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_mismatched_participant_is_decisive(params):
    """Confidence law: 'John purchased a car' vs 'John purchased a
    truck' — the matched subject path cannot rescue the statement pair.
    Negative evidence has full weight while positive evidence is shrunk
    by corroboration count, so the mismatched object drags the pair
    below the neutral prior under every variant. The same-name subjects
    are measurably touched: their confidence strictly drops below its
    value in the matched-object corpus.

    Domain discovery: the shared subject has no gate. Its confidence
    settles at (1 − na) · seed, which stays above low thresholds (the
    Johns merge at threshold 0.1) and under weak saturation stays above
    the neutral prior itself — so the lawful form of the original 'Johns
    do not merge' assertion is this confidence ordering, not a merge
    claim."""
    mismatched, johns, purchases = _purchase_corpus(["car", "truck"])
    clean, clean_johns, _ = _purchase_corpus(["car", "car"])

    confidence, _, _ = match_graphs(mismatched, params=frozen(params))
    clean_confidence, _, _ = match_graphs(clean, params=frozen(params))

    assert confidence[key(purchases[0], purchases[1])] < NEUTRAL_PRIOR
    john_pair = key(johns[0], johns[1])
    assert confidence[john_pair] < clean_confidence[key(clean_johns[0], clean_johns[1])]


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_hub_participants_weaken_slot_evidence(params):
    """Confidence law: a participant appearing in many statements is a
    weak identity signal — inverse functionality dilution. The same
    acquire/purchase pair scores strictly lower when both participants
    are hubs (each in six further background statements) than in the
    clean two-graph corpus, under every variant."""
    clean = _deal_corpus(["acquire", "purchase"])
    hub = _deal_corpus(["acquire", "purchase"])
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
        bg = Graph(id=f"bg{i}")
        bg.add_statement(
            bg.add_entity("Acme Corp"), predicate, bg.add_entity("Beta Systems")
        )
        hub.graphs.append(bg)

    conf_clean, _, _ = match_graphs(clean.graphs, params=frozen(params))
    conf_hub, _, _ = match_graphs(hub.graphs, params=frozen(params))

    pair = key(hub.deals[0], hub.deals[1])
    assert conf_hub[pair] < conf_clean[key(clean.deals[0], clean.deals[1])]


def test_participant_merge_never_fabricates_negative_evidence():
    """Regression law: a mid-run merge of participants must not turn
    into decisive negative evidence against a still-unmerged statement
    pair — adjacency staleness after a merge is not evidence. With
    shrinkage strong (prior_strength 10) the paraphrase pair rests well
    below the bar while the name-identical participants merge mid-run
    (their pinned 1.0 confidence commits at any threshold <= 1). The
    statement pair must keep its fixed point: after the participant
    merge both statements see the same canonical participant, which if
    anything strengthens the pair — its confidence is at least the
    frozen-run value and stays above the neutral prior."""
    corpus = _deal_corpus(["acquire", "purchase"])

    merging = CALIBRATION._replace(prior_strength=10.0)
    frozen_run = merging._replace(merge_threshold=1.01)

    conf_merging, groups_merging, _ = match_graphs(corpus.graphs, params=merging)
    conf_frozen, _, _ = match_graphs(corpus.graphs, params=frozen_run)

    # The scenario holds: participants merged mid-run, statements did not.
    assert same_group(groups_merging, *corpus.subjects)
    assert same_group(groups_merging, *corpus.objects)
    assert not same_group(groups_merging, *corpus.deals)

    pair = key(corpus.deals[0], corpus.deals[1])
    assert conf_merging[pair] >= conf_frozen[pair]
    assert conf_merging[pair] > NEUTRAL_PRIOR


# ---------------------------------------------------------------------------
# Can-merge laws: capability under explicit extremes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predicate_prior",
    [None, _acquire_purchase_prior],
    ids=["neutral-prior", "hot-prior"],
)
def test_paraphrase_can_merge(predicate_prior):
    """Can-merge law: the machinery is capable of merging a paraphrase
    pair on slot-aligned identical participants alone — loose parameters
    (threshold barely above the neutral prior, weak shrinkage), with or
    without a hot predicate prior proposing the pair. Capability, not
    calibration: whether it merges under CALIBRATION is evals'
    question."""
    corpus = _deal_corpus(["acquire", "purchase"])

    _, groups, _ = match_graphs(
        corpus.graphs, params=LOOSE, predicate_prior=predicate_prior
    )

    assert same_group(groups, *corpus.deals)
    assert same_group(groups, *corpus.subjects)
    assert same_group(groups, *corpus.objects)


def test_third_witness_can_join_transitively():
    """Can-merge law: three outlets reporting one deal with three
    predicate wordings can all land in one group of three statements —
    union-find commits the transitive closure as pairwise lifts cross
    the (loose) bar."""
    corpus = _deal_corpus(["acquire", "purchase", "take over"])

    _, groups, _ = match_graphs(corpus.graphs, params=LOOSE)

    assert same_group(groups, *corpus.deals)
    assert same_group(groups, *corpus.subjects)


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_unreachable_threshold_merges_nothing(params):
    """Can-merge law (negative extreme): with the threshold above 1 —
    unreachable by any confidence — nothing merges, however strong the
    structure. The depth-2 nested corpus (two reports of a deal, its
    denial, and a wire report of the denial) has seven unambiguous term
    pairs and produces zero groups under every variant."""
    graphs = []
    for i, predicate in enumerate(["acquire", "purchase"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        beta = g.add_entity("Beta Systems")
        deal = g.add_statement(acme, predicate, beta)
        spokesperson = g.add_entity("Company Spokesperson")
        denial = g.add_statement(spokesperson, "deny", deal)
        agency = g.add_entity("Meridian Wire")
        g.add_statement(agency, "report", denial)
        graphs.append(g)

    _, groups, _ = match_graphs(graphs, params=frozen(params))

    assert groups == []


# ---------------------------------------------------------------------------
# Recursion is free: statements about statements match through the same
# slots as any participant (can-merge laws)
# ---------------------------------------------------------------------------


def test_shared_rare_qualifier_can_corroborate():
    """Can-merge law (nesting through the subject slot): two reports of
    one deal share the slot-aligned price qualifier (deal, 'for',
    '$4.2 billion') — a rare periphery participant. Under loose
    parameters the qualifier statements can merge, lifted through their
    subject slots by the deal pair, which merges on its own
    participants; the deal, its qualifier, and the price all land in one
    group each. Positive mirror of the HQ disagreement: a shared rare
    participant corroborates where a mismatched one contradicts."""
    graphs, deals, qualifiers, prices = [], [], [], []
    for i, predicate in enumerate(["acquire", "purchase"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        beta = g.add_entity("Beta Systems")
        deal = g.add_statement(acme, predicate, beta)
        price = g.add_entity("$4.2 billion")
        qualifier = g.add_statement(deal, "for", price)
        deals.append(deal)
        qualifiers.append(qualifier)
        prices.append(price)
        graphs.append(g)

    _, groups, _ = match_graphs(graphs, params=LOOSE)

    assert same_group(groups, *deals)
    assert same_group(groups, *qualifiers)
    assert same_group(groups, *prices)


def test_denial_of_the_same_claim_can_merge():
    """Can-merge law (nesting through the object slot): (spokesperson,
    'deny', deal) in two reports of the same deal. The denials can
    merge under loose parameters, lifted through their object slots by
    the deal pair beneath them — which merges on its own participants.
    Attribution and denial need no special machinery: a statement
    participant is just a participant."""
    graphs, deals, denials = [], [], []
    for i, predicate in enumerate(["acquire", "purchase"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        beta = g.add_entity("Beta Systems")
        deal = g.add_statement(acme, predicate, beta)
        spokesperson = g.add_entity("Company Spokesperson")
        denial = g.add_statement(spokesperson, "deny", deal)
        deals.append(deal)
        denials.append(denial)
        graphs.append(g)

    _, groups, _ = match_graphs(graphs, params=LOOSE)

    assert same_group(groups, *deals)
    assert same_group(groups, *denials)


def test_nesting_two_levels_deep_can_merge():
    """Can-merge law: a claim, its denial, and a wire report of the
    denial — (agency, 'report', (spokesperson, 'deny', (acme, 'acquire',
    beta))) — every nesting level can merge under loose parameters, each
    layer lifted through the same slot evidence as the layer beneath it.
    Recursion costs nothing: there is one mechanism, applied to terms."""
    graphs, deals, denials, reports = [], [], [], []
    for i, predicate in enumerate(["acquire", "purchase"]):
        g = Graph(id=f"g{i}")
        acme = g.add_entity("Acme Corp")
        beta = g.add_entity("Beta Systems")
        deal = g.add_statement(acme, predicate, beta)
        spokesperson = g.add_entity("Company Spokesperson")
        denial = g.add_statement(spokesperson, "deny", deal)
        agency = g.add_entity("Meridian Wire")
        report = g.add_statement(agency, "report", denial)
        deals.append(deal)
        denials.append(denial)
        reports.append(report)
        graphs.append(g)

    _, groups, _ = match_graphs(graphs, params=LOOSE)

    assert same_group(groups, *deals)
    assert same_group(groups, *denials)
    assert same_group(groups, *reports)


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
    confidence, _, _ = match_graphs(_scenario(), params=CALIBRATION)

    assert confidence
    for (a, b), score in confidence.items():
        assert (b, a) in confidence
        assert confidence[(b, a)] == score


def test_graph_order_does_not_change_results():
    """Permutation invariance: the input order of graphs cannot change
    groups or confidences."""
    import itertools

    graphs = _scenario()
    conf_ref, groups_ref, _ = match_graphs(graphs, params=CALIBRATION)
    ref = {frozenset(g) for g in groups_ref}

    for permutation in itertools.permutations(graphs):
        conf, groups, _ = match_graphs(list(permutation), params=CALIBRATION)
        assert {frozenset(g) for g in groups} == ref
        assert conf == conf_ref


def test_identical_runs_are_deterministic():
    graphs = _scenario()
    conf_a, groups_a, merged_a = match_graphs(graphs, params=CALIBRATION)
    conf_b, groups_b, merged_b = match_graphs(graphs, params=CALIBRATION)

    assert conf_a == conf_b
    assert groups_a == groups_b
    assert merged_a.terms.keys() == merged_b.terms.keys()


def test_single_graph_matches_nothing():
    """No cross-graph pairs exist for a single graph."""
    g = Graph(id="g1")
    acme = g.add_entity("Acme Corp")
    g.add_statement(acme, "acquire", g.add_entity("Beta Systems"))

    confidence, groups, merged = match_graphs([g], params=CALIBRATION)

    assert confidence == {}
    assert groups == []
    assert len(merged.terms) == 3


def test_no_graphs_is_empty():
    confidence, groups, merged = match_graphs([], params=CALIBRATION)

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
        match_graphs([g1, g2], params=CALIBRATION)
