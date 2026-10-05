"""Scenario corpus for the matcher eval: a ladder of tiny corpora.

Each scenario is a smallest-corpus exhibit of one rung of the merge
boundary, with focal term pairs tagged by intended outcome: ``merge``
and ``apart`` are claims about what the calibration should do;
``boundary`` marks pairs whose correct fate is genuinely unknowable at
that corpus size (two articles, one statement each — name co-occurrence
is weak evidence, structure uninformative), observed and reported, never
judged. Unsatisfied intents are findings about the parameter values, not
failures of the harness.

Statement pairs seed via the neutral prior (``predicate_prior=None``)
except where a rung pins a gate with a deterministic stub prior. Graphs
are built with explicit term ids (fixed per graph namespace) so focal
pairs are static; ``match_graphs`` is a pure consumer, so one
construction serves every grid point.
"""

from dataclasses import dataclass
from typing import Literal

from worldgraph.graph import Graph, Statement
from worldgraph.match import PredicatePrior, Qid

Intent = Literal["merge", "apart", "boundary"]

# Tags that carry claims (boundary is observation-only).
CLAIMED_INTENTS = ("merge", "apart")


@dataclass(frozen=True)
class FocalPair:
    """One cross-graph term pair the eval watches.

    ``label`` is display-only; identity is the qualified-id pair.
    """

    label: str
    left: Qid
    right: Qid
    intent: Intent


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    graphs: tuple[Graph, ...]
    focal_pairs: tuple[FocalPair, ...]
    predicate_prior: PredicatePrior | None = None


def _deal_graph(
    graph_id: str,
    predicate: str,
    *,
    buyer: str = "Acme Corp",
    target: str = "Beta Systems",
) -> Graph:
    """Smallest statement corpus: buyer —predicate→ target, with fixed
    term ids (buyer/target/deal) so focal pairs are uniform across
    scenarios."""
    graph = Graph(id=graph_id)
    buyer_term = graph.add_entity(buyer, id="buyer")
    target_term = graph.add_entity(target, id="target")
    graph.add_statement(buyer_term, predicate, target_term, id="deal")
    return graph


def _gate_no_seeds() -> Scenario:
    """Structure gate on the statement side: a hot prior proposes, failed
    counterpart searches dispose."""
    g1 = Graph(id="g1")
    stellar1 = g1.add_entity("Stellar Foods", id="stellar")
    nova1 = g1.add_entity("Nova Markets", id="nova")
    g1.add_statement(stellar1, "acquire", nova1, id="acquire")
    orbit1 = g1.add_entity("Orbit Logistics", id="orbit")
    port1 = g1.add_entity("Eastport", id="port")
    g1.add_statement(orbit1, "lease", port1, id="lease")

    g2 = Graph(id="g2")
    canyon2 = g2.add_entity("Canyon Media", id="canyon")
    pulse2 = g2.add_entity("Pulse Radio", id="pulse")
    g2.add_statement(canyon2, "acquire", pulse2, id="acquire")
    quay2 = g2.add_entity("Quay Partners", id="quay")
    dock2 = g2.add_entity("Northdock", id="dock")
    g2.add_statement(quay2, "lease", dock2, id="lease")

    def hot_prior(_a: Statement, _b: Statement) -> float:
        return 0.95

    return Scenario(
        id="gate-no-seeds",
        description=(
            "Two two-statement corpora whose participants share no names. A hot "
            "0.95 predicate prior proposes both statement pairs, but every "
            "slot-aligned counterpart search fails — no cross-graph name seeds "
            "— so each proposal meets decisive negative evidence. Intent: both "
            "statement pairs stay apart at every parameter vector; this rung "
            "pins 'similarity proposes, structure disposes'."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair(
                "acquire statements", ("g1", "acquire"), ("g2", "acquire"), "apart"
            ),
            FocalPair("lease statements", ("g1", "lease"), ("g2", "lease"), "apart"),
        ),
        predicate_prior=hot_prior,
    )


def _gate_lone_names() -> Scenario:
    """Structure gate on the entity side: a 1.0 name seed with nothing to
    test can never merge."""
    g1 = Graph(id="g1")
    g1.add_entity("Acme Corp", id="acme")
    g2 = Graph(id="g2")
    g2.add_entity("Acme Corp", id="acme")

    return Scenario(
        id="gate-lone-names",
        description=(
            "Identical lone entity names ('Acme Corp') in two graphs with no "
            "statements. The name seed is 1.0, but there is no structure to "
            "test: n_tested stays 0 and the pair can never merge, whatever the "
            "threshold — too many John Smiths for a bare name to carry. The "
            "report shows confidence pinned at 1.0 across the whole grid with "
            "zero merges: the gate, not the bar, holds the pair apart."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("lone entities", ("g1", "acme"), ("g2", "acme"), "apart"),
        ),
    )


def _duplicate_minimal() -> Scenario:
    """Two byte-identical single-statement articles: the boundary itself."""
    g1 = _deal_graph("g1", "acquire")
    g2 = _deal_graph("g2", "acquire")

    return Scenario(
        id="duplicate-minimal",
        description=(
            "Two IDENTICAL single-statement articles: same predicate, identical "
            "participant names. Whether two such tiny articles should merge is "
            "genuinely unknowable — name co-occurrence is weak evidence and "
            "structure is uninformative at this size — so the eval observes, "
            "and the report shows where on the grid the boundary falls. Under "
            "CALIBRATION all three pairs merge."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("deal statements", ("g1", "deal"), ("g2", "deal"), "boundary"),
            FocalPair("buyer entities", ("g1", "buyer"), ("g2", "buyer"), "boundary"),
            FocalPair(
                "target entities", ("g1", "target"), ("g2", "target"), "boundary"
            ),
        ),
    )


def _paraphrase_minimal() -> Scenario:
    """acquire/purchase over identical participants: the same boundary
    question, wording variation included."""
    g1 = _deal_graph("g1", "acquire")
    g2 = _deal_graph("g2", "purchase")

    return Scenario(
        id="paraphrase-minimal",
        description=(
            "acquire vs purchase over identical participants, neutral prior. "
            "Predicates are never compared as text, so under the neutral prior "
            "this corpus is computationally identical to duplicate-minimal — "
            "the report's identical rows make that visible. Boundary: whether "
            "two one-statement articles merge is the empirical question this "
            "eval exists to expose, not assert."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("deal statements", ("g1", "deal"), ("g2", "deal"), "boundary"),
            FocalPair("buyer entities", ("g1", "buyer"), ("g2", "buyer"), "boundary"),
            FocalPair(
                "target entities", ("g1", "target"), ("g2", "target"), "boundary"
            ),
        ),
    )


def _unrelated_minimal() -> Scenario:
    """acquire vs compete with over identical participants: an intent the
    neutral prior cannot deliver — the finding, not a harness bug."""
    g1 = _deal_graph("g1", "acquire")
    g2 = _deal_graph("g2", "compete with")

    return Scenario(
        id="unrelated-minimal",
        description=(
            "acquire vs compete with over identical participants, neutral "
            "prior. Intent: these are different facts — the statements must "
            "stay apart. The trap: the propagation core compares no text, so "
            "under the neutral prior this corpus is structurally "
            "indistinguishable from paraphrase-minimal; separating them is the "
            "predicate prior's job, not structure's. The intent is expected to "
            "be unsatisfied in much of the grid — an unsatisfied intent is a "
            "finding about the calibration. The entity pairs are boundary: "
            "same companies, one statement each, disputed facts."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("deal statements", ("g1", "deal"), ("g2", "deal"), "apart"),
            FocalPair("buyer entities", ("g1", "buyer"), ("g2", "buyer"), "boundary"),
            FocalPair(
                "target entities", ("g1", "target"), ("g2", "target"), "boundary"
            ),
        ),
    )


def _qualifier_corroboration() -> Scenario:
    """A shared rare qualifier corroborates two paraphrased deals."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp", id="buyer")
    beta1 = g1.add_entity("Beta Systems", id="target")
    deal1 = g1.add_statement(acme1, "acquire", beta1, id="deal")
    price1 = g1.add_entity("$4.2 billion", id="price")
    g1.add_statement(deal1, "for", price1, id="price-stmt")

    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp", id="buyer")
    beta2 = g2.add_entity("Beta Systems", id="target")
    deal2 = g2.add_statement(acme2, "purchase", beta2, id="deal")
    price2 = g2.add_entity("$4.2 billion", id="price")
    g2.add_statement(deal2, "for", price2, id="price-stmt")

    return Scenario(
        id="qualifier-corroboration",
        description=(
            "Two reports of one deal (acquire/purchase) that additionally share "
            "a slot-aligned rare price qualifier ($4.2 billion). The qualifier "
            "corroborates the deal pair through its subject slot and lifts in "
            "turn through the merged deal — nesting is free. Intent: deal "
            "statements, price statements, and price entities all merge."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("deal statements", ("g1", "deal"), ("g2", "deal"), "merge"),
            FocalPair(
                "price statements", ("g1", "price-stmt"), ("g2", "price-stmt"), "merge"
            ),
            FocalPair("price entities", ("g1", "price"), ("g2", "price"), "merge"),
        ),
    )


def _three_witnesses() -> Scenario:
    """Three outlets, three predicates, one deal."""
    graphs = tuple(
        _deal_graph(f"g{i}", predicate)
        for i, predicate in enumerate(["acquire", "purchase", "take over"])
    )

    return Scenario(
        id="three-witnesses",
        description=(
            "Three outlets report the same deal with three predicates "
            "(acquire/purchase/take over) over identical participants. Three "
            "witnesses is enough corroboration that the outcome should no "
            "longer be boundary. Intent: every statement pair merges into one "
            "group of three."
        ),
        graphs=graphs,
        focal_pairs=(
            FocalPair("deal g0–g1", ("g0", "deal"), ("g1", "deal"), "merge"),
            FocalPair("deal g0–g2", ("g0", "deal"), ("g2", "deal"), "merge"),
            FocalPair("deal g1–g2", ("g1", "deal"), ("g2", "deal"), "merge"),
        ),
    )


def _mismatch_twin() -> Scenario:
    """Matched subject, mismatched object: negative evidence is decisive."""
    g1 = Graph(id="g1")
    john1 = g1.add_entity("John Mercer", id="john")
    car1 = g1.add_entity("car", id="vehicle")
    g1.add_statement(john1, "purchase", car1, id="purchase")

    g2 = Graph(id="g2")
    john2 = g2.add_entity("John Mercer", id="john")
    truck2 = g2.add_entity("truck", id="vehicle")
    g2.add_statement(john2, "purchase", truck2, id="purchase")

    return Scenario(
        id="mismatch-twin",
        description=(
            "John Mercer purchased a car / John Mercer purchased a truck: "
            "matched subject, mismatched object. One mismatched participant is "
            "decisive — negative evidence needs no corroboration. Intent: the "
            "statements stay apart AND the same-name Johns stay apart (their "
            "only structural interaction is the suppressed statement)."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair(
                "purchase statements", ("g1", "purchase"), ("g2", "purchase"), "apart"
            ),
            FocalPair("John Mercer entities", ("g1", "john"), ("g2", "john"), "apart"),
        ),
    )


def _direction_flip() -> Scenario:
    """(council governs district) vs (district governs council)."""
    g1 = Graph(id="g1")
    council1 = g1.add_entity("City Council", id="council")
    district1 = g1.add_entity("Harbor District", id="district")
    g1.add_statement(council1, "govern", district1, id="govern")

    g2 = Graph(id="g2")
    council2 = g2.add_entity("City Council", id="council")
    district2 = g2.add_entity("Harbor District", id="district")
    g2.add_statement(district2, "govern", council2, id="govern")

    return Scenario(
        id="direction-flip",
        description=(
            "(City Council governs Harbor District) vs (Harbor District "
            "governs City Council): identical names, identical predicate, "
            "opposite direction. Slot alignment makes each counterpart search "
            "land on the wrong side — negative evidence, never a merge — and "
            "the name-identical entities find no slot-matched counterpart at "
            "all. Intent: the statements and both entity pairs stay apart."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("govern statements", ("g1", "govern"), ("g2", "govern"), "apart"),
            FocalPair(
                "council entities", ("g1", "council"), ("g2", "council"), "apart"
            ),
            FocalPair(
                "district entities", ("g1", "district"), ("g2", "district"), "apart"
            ),
        ),
    )


_HUB_PREDICATES = (
    "sign agreement with",
    "partner with",
    "jointly develop",
    "co-market",
    "form alliance with",
    "share premises with",
)


def _hub_clean() -> Scenario:
    """Reference arm of the hub pair: the deal in a clean corpus."""
    graphs = (
        _deal_graph("hub-clean-1", "acquire", buyer="Acme Corp", target="Gamma AI"),
        _deal_graph("hub-clean-2", "purchase", buyer="Acme Corp", target="Gamma AI"),
    )

    return Scenario(
        id="hub-clean",
        description=(
            "Reference arm of the hub pair: an acquire/purchase deal over "
            "name-identical participants in a clean two-graph corpus. Intent: "
            "the statements merge. Compare hub-hubbed, the treatment arm."
        ),
        graphs=graphs,
        focal_pairs=(
            FocalPair(
                "deal statements",
                ("hub-clean-1", "deal"),
                ("hub-clean-2", "deal"),
                "merge",
            ),
        ),
    )


def _hub_hubbed() -> Scenario:
    """Treatment arm: the same deal with both participants hubbed by six
    background statements each."""
    graphs = [
        _deal_graph("hub-hubbed-1", "acquire", buyer="Acme Corp", target="Gamma AI"),
        _deal_graph("hub-hubbed-2", "purchase", buyer="Acme Corp", target="Gamma AI"),
    ]
    graphs.extend(
        _deal_graph(f"hub-bg{i}", predicate, buyer="Acme Corp", target="Gamma AI")
        for i, predicate in enumerate(_HUB_PREDICATES)
    )

    return Scenario(
        id="hub-hubbed",
        description=(
            "Treatment arm of the hub pair: the same acquire/purchase deal, "
            "but both participants are hubbed — six further graphs reuse the "
            "same names, so each participant pools eight statements and its "
            "inverse functionality drops: a participant appearing in many "
            "statements is a weak identity signal. Intent: the statements stay "
            "apart. (Run as its own corpus because functionality pools "
            "participants by name across the whole corpus — a single corpus "
            "cannot hold both arms without contaminating them.)"
        ),
        graphs=tuple(graphs),
        focal_pairs=(
            FocalPair(
                "deal statements",
                ("hub-hubbed-1", "deal"),
                ("hub-hubbed-2", "deal"),
                "apart",
            ),
        ),
    )


def _partial_merge_ordering() -> Scenario:
    """Statement confidence when entities merge first: name variants put
    entity seeds below 1.0."""
    g1 = _deal_graph("g1", "acquire")
    g2 = _deal_graph(
        "g2", "purchase", buyer="Acme Corporation", target="Beta Systems Ltd"
    )

    return Scenario(
        id="partial-merge-ordering",
        description=(
            "acquire/purchase over participants whose names differ slightly "
            "(Acme Corp/Acme Corporation, Beta Systems/Beta Systems Ltd), so "
            "the entity seeds sit below 1.0 and the statement pair's fate "
            "depends on merge order: until the entity pairs merge, each "
            "statement sees two distinct counterpart candidates per slot; "
            "after, one canonical participant. The observable is the statement "
            "pair's confidence across the grid — where it clears the bar only "
            "through entities that merged first. Boundary, with emphasis on "
            "the margin column: how far the statement rides on its "
            "participants' merge."
        ),
        graphs=(g1, g2),
        focal_pairs=(
            FocalPair("deal statements", ("g1", "deal"), ("g2", "deal"), "boundary"),
        ),
    )


SCENARIOS: tuple[Scenario, ...] = (
    _gate_no_seeds(),
    _gate_lone_names(),
    _duplicate_minimal(),
    _paraphrase_minimal(),
    _unrelated_minimal(),
    _qualifier_corroboration(),
    _three_witnesses(),
    _mismatch_twin(),
    _direction_flip(),
    _hub_clean(),
    _hub_hubbed(),
    _partial_merge_ordering(),
)
