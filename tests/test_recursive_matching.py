"""Public-interface tests for matching recursively attached edge occurrences."""

from worldgraph.graph import Graph
from worldgraph.match import match_graphs


def _groups(*groups):
    return {frozenset(term.id for term in group) for group in groups}


def _assert_original_occurrences(unified, *graphs):
    expected_nodes = {
        node.id: node for graph in graphs for node in graph.nodes.values()
    }
    expected_edges = {
        edge.id: edge for graph in graphs for edge in graph.edges.values()
    }

    assert unified.id == "unified"
    assert unified.nodes == expected_nodes
    assert unified.edges == expected_edges


def test_qualifier_and_qualified_fact_match_as_edge_occurrences(embedder):
    g1 = Graph(id="article-1")
    acme1 = g1.add_entity("Acme Corp")
    datavault1 = g1.add_entity("DataVault")
    price1 = g1.add_entity("$4 billion")
    purchase1 = g1.add_edge(acme1, datavault1, "acquire")
    valuation1 = g1.add_edge(purchase1, price1, "value at")

    g2 = Graph(id="article-2")
    acme2 = g2.add_entity("Acme Corp")
    datavault2 = g2.add_entity("DataVault")
    price2 = g2.add_entity("$4 billion")
    purchase2 = g2.add_edge(acme2, datavault2, "purchase")
    valuation2 = g2.add_edge(purchase2, price2, "price at")

    _, match_groups, unified = match_graphs([g1, g2], embedder)

    assert {frozenset(group) for group in match_groups} == _groups(
        (acme1, acme2),
        (datavault1, datavault2),
        (price1, price2),
        (purchase1, purchase2),
        (valuation1, valuation2),
    )
    _assert_original_occurrences(unified, g1, g2)


def test_edge_used_as_target_matches_at_both_levels(embedder):
    g1 = Graph(id="article-1")
    marisol1 = g1.add_entity("Marisol Vaneck")
    vesterby1 = g1.add_entity("Vesterby")
    ivo1 = g1.add_entity("Ivo Brandt")
    visit1 = g1.add_edge(marisol1, vesterby1, "visit")
    participation1 = g1.add_edge(ivo1, visit1, "join")

    g2 = Graph(id="article-2")
    marisol2 = g2.add_entity("Marisol Vaneck")
    vesterby2 = g2.add_entity("Vesterby")
    ivo2 = g2.add_entity("Ivo Brandt")
    visit2 = g2.add_edge(marisol2, vesterby2, "travel to")
    participation2 = g2.add_edge(ivo2, visit2, "take part in")

    _, match_groups, unified = match_graphs([g1, g2], embedder)

    assert {frozenset(group) for group in match_groups} == _groups(
        (marisol1, marisol2),
        (vesterby1, vesterby2),
        (ivo1, ivo2),
        (visit1, visit2),
        (participation1, participation2),
    )
    _assert_original_occurrences(unified, g1, g2)


def test_depth_two_edge_attachment_chain_matches_at_every_level(embedder):
    g1 = Graph(id="article-1")
    lena1 = g1.add_entity("Lena Orvik")
    torshavn1 = g1.add_entity("Torshavn")
    pavel1 = g1.add_entity("Pavel Rusk")
    council1 = g1.add_entity("Norvale Council")
    visit1 = g1.add_edge(lena1, torshavn1, "visit")
    participation1 = g1.add_edge(pavel1, visit1, "join")
    representation1 = g1.add_edge(participation1, council1, "on behalf of")

    g2 = Graph(id="article-2")
    lena2 = g2.add_entity("Lena Orvik")
    torshavn2 = g2.add_entity("Torshavn")
    pavel2 = g2.add_entity("Pavel Rusk")
    council2 = g2.add_entity("Norvale Council")
    visit2 = g2.add_edge(lena2, torshavn2, "travel to")
    participation2 = g2.add_edge(pavel2, visit2, "take part in")
    representation2 = g2.add_edge(participation2, council2, "represent")

    _, match_groups, unified = match_graphs([g1, g2], embedder)

    assert {frozenset(group) for group in match_groups} == _groups(
        (lena1, lena2),
        (torshavn1, torshavn2),
        (pavel1, pavel2),
        (council1, council2),
        (visit1, visit2),
        (participation1, participation2),
        (representation1, representation2),
    )
    _assert_original_occurrences(unified, g1, g2)


def test_attachment_selects_the_corresponding_edge_occurrence(embedder):
    g1 = Graph(id="article-1")
    acme1 = g1.add_entity("Acme Corp")
    datavault1 = g1.add_entity("DataVault")
    price1 = g1.add_entity("$4 billion")
    purchase1 = g1.add_edge(acme1, datavault1, "acquire")
    valuation1 = g1.add_edge(purchase1, price1, "value at")

    g2 = Graph(id="article-2")
    acme2 = g2.add_entity("Acme Corp")
    datavault2 = g2.add_entity("DataVault")
    price2 = g2.add_entity("$4 billion")
    qualified_purchase2 = g2.add_edge(acme2, datavault2, "purchase")
    other_purchase2 = g2.add_edge(acme2, datavault2, "purchase")
    valuation2 = g2.add_edge(qualified_purchase2, price2, "price at")

    _, match_groups, unified = match_graphs([g1, g2], embedder)

    assert {frozenset(group) for group in match_groups} == _groups(
        (acme1, acme2),
        (datavault1, datavault2),
        (price1, price2),
        (purchase1, qualified_purchase2),
        (valuation1, valuation2),
    )
    assert all(other_purchase2.id not in group for group in match_groups)
    _assert_original_occurrences(unified, g1, g2)


def test_relation_to_relation_on_both_endpoints_matches_recursively(embedder):
    g1 = Graph(id="article-1")
    halden1 = g1.add_entity("Halden Energy")
    station1 = g1.add_entity("Vesterby Power Station")
    meridian1 = g1.add_entity("Meridian Rail")
    kalden1 = g1.add_entity("Kalden")
    closure1 = g1.add_edge(halden1, station1, "close")
    suspension1 = g1.add_edge(meridian1, kalden1, "suspend services to")
    cause1 = g1.add_edge(closure1, suspension1, "cause")

    g2 = Graph(id="article-2")
    halden2 = g2.add_entity("Halden Energy")
    station2 = g2.add_entity("Vesterby Power Station")
    meridian2 = g2.add_entity("Meridian Rail")
    kalden2 = g2.add_entity("Kalden")
    closure2 = g2.add_edge(halden2, station2, "shut down")
    suspension2 = g2.add_edge(meridian2, kalden2, "halt services to")
    cause2 = g2.add_edge(closure2, suspension2, "result in")

    _, match_groups, unified = match_graphs([g1, g2], embedder)

    assert {frozenset(group) for group in match_groups} == _groups(
        (halden1, halden2),
        (station1, station2),
        (meridian1, meridian2),
        (kalden1, kalden2),
        (closure1, closure2),
        (suspension1, suspension2),
        (cause1, cause2),
    )
    _assert_original_occurrences(unified, g1, g2)
