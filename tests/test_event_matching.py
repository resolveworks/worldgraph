"""Public-interface tests for matching event nodes across graphs.

Events are unseeded nodes: their labels are never compared, so event
matches are decided purely by role-aligned participant structure.
Qualifiers are patients of the event they qualify; events may participate
in other events (joining a visit, causing a suspension).
"""


from conftest import fact

from worldgraph.graph import Graph
from worldgraph.match import match_graphs


def _groups(*groups):
    return {frozenset(node.id for node in group) for group in groups}


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


def test_identical_facts_match_entities_and_event():
    """Same names, same label: entities and the event all merge."""
    g1 = Graph(id="article-1")
    acme1 = g1.add_entity("Acme Corp")
    gamma1 = g1.add_entity("Gamma AI")
    acquisition1 = fact(g1, "acquire", agent=acme1, patients=(gamma1,))

    g2 = Graph(id="article-2")
    acme2 = g2.add_entity("Acme Corp")
    gamma2 = g2.add_entity("Gamma AI")
    acquisition2 = fact(g2, "acquire", agent=acme2, patients=(gamma2,))

    _, match_groups, unified = match_graphs([g1, g2])

    assert {frozenset(group) for group in match_groups} == _groups(
        (acme1, acme2),
        (gamma1, gamma2),
        (acquisition1, acquisition2),
    )
    _assert_original_occurrences(unified, g1, g2)


def test_synonymous_labels_match_when_participants_match():
    """'acquire' vs 'take control of': labels are never compared — the
    events merge on aligned agent and patient alone."""
    g1 = Graph(id="article-1")
    buyer1 = g1.add_entity("Acme Corp")
    target1 = g1.add_entity("Gamma AI")
    acquisition = fact(g1, "acquire", agent=buyer1, patients=(target1,))

    g2 = Graph(id="article-2")
    buyer2 = g2.add_entity("Acme Corp")
    target2 = g2.add_entity("Gamma AI")
    takeover = fact(g2, "take control of", agent=buyer2, patients=(target2,))

    _, groups, _ = match_graphs([g1, g2])

    assert {frozenset(group) for group in groups} == _groups(
        (buyer1, buyer2),
        (target1, target2),
        (acquisition, takeover),
    )


def test_swapped_roles_do_not_match():
    """'Acme acquires Gamma' vs 'Gamma acquires Acme': the same entities in
    swapped roles produce no role-aligned paths — and negative evidence
    from the mismatched roles keeps even the same-name entities apart."""
    g1 = Graph(id="article-1")
    acme1 = g1.add_entity("Acme Corp")
    gamma1 = g1.add_entity("Gamma AI")
    acquisition1 = fact(g1, "acquire", agent=acme1, patients=(gamma1,))

    g2 = Graph(id="article-2")
    gamma2 = g2.add_entity("Gamma AI")
    acme2 = g2.add_entity("Acme Corp")
    acquisition2 = fact(g2, "acquire", agent=gamma2, patients=(acme2,))

    _, groups, _ = match_graphs([g1, g2])

    assert groups == [], f"swapped roles merged: {groups}"
    assert acquisition1.id != acquisition2.id


def test_unrelated_participants_do_not_match():
    """Same label, entirely different participants: no match."""
    g1 = Graph(id="article-1")
    acme1 = g1.add_entity("Acme Corp")
    gamma1 = g1.add_entity("Gamma AI")
    fact(g1, "acquire", agent=acme1, patients=(gamma1,))

    g2 = Graph(id="article-2")
    north2 = g2.add_entity("Northstar Labs")
    cloud2 = g2.add_entity("CloudScale")
    fact(g2, "acquire", agent=north2, patients=(cloud2,))

    _, groups, _ = match_graphs([g1, g2])

    assert groups == []


def test_qualified_event_matches_despite_paraphrase():
    """Qualifiers are patients of the event they qualify. 'manage' with
    role and scope qualifiers matches its paraphrase 'oversee' because all
    three participants align."""
    g1 = Graph(id="article-1")
    tessa1 = g1.add_entity("Tessa Corin")
    halden1 = g1.add_entity("Halden Freight")
    vesterby1 = g1.add_entity("Vesterby")
    director1 = g1.add_entity("managing director")
    manage1 = fact(
        g1, "manage", agent=tessa1, patients=(halden1, director1, vesterby1)
    )

    g2 = Graph(id="article-2")
    tessa2 = g2.add_entity("Tessa Corin")
    halden2 = g2.add_entity("Halden Freight")
    vesterby2 = g2.add_entity("Vesterby")
    director2 = g2.add_entity("managing director")
    manage2 = fact(
        g2, "oversee", agent=tessa2, patients=(halden2, director2, vesterby2)
    )

    _, groups, _ = match_graphs([g1, g2])

    assert {frozenset(group) for group in groups} == _groups(
        (tessa1, tessa2),
        (halden1, halden2),
        (vesterby1, vesterby2),
        (director1, director2),
        (manage1, manage2),
    )


def test_nested_events_match_recursively():
    """Events participating in events: Ivo joins Marisol's visit. The join
    events match only once the visit events match, which match only once
    the participants match — recursion through the same mechanism."""
    g1 = Graph(id="article-1")
    marisol1 = g1.add_entity("Marisol Vaneck")
    vesterby1 = g1.add_entity("Vesterby")
    ivo1 = g1.add_entity("Ivo Brandt")
    visit1 = fact(g1, "visit", agent=marisol1, patients=(vesterby1,))
    participation1 = fact(g1, "join", agent=ivo1, patients=(visit1,))

    g2 = Graph(id="article-2")
    marisol2 = g2.add_entity("Marisol Vaneck")
    vesterby2 = g2.add_entity("Vesterby")
    ivo2 = g2.add_entity("Ivo Brandt")
    visit2 = fact(g2, "travel to", agent=marisol2, patients=(vesterby2,))
    participation2 = fact(g2, "take part in", agent=ivo2, patients=(visit2,))

    _, groups, unified = match_graphs([g1, g2])

    assert {frozenset(group) for group in groups} == _groups(
        (marisol1, marisol2),
        (vesterby1, vesterby2),
        (ivo1, ivo2),
        (visit1, visit2),
        (participation1, participation2),
    )
    _assert_original_occurrences(unified, g1, g2)


def test_causation_between_events_matches_on_both_levels():
    """A 'cause' event whose agent and patient are themselves events:
    closure causes suspension. Matches at the event level on both sides
    of the causal link."""
    g1 = Graph(id="article-1")
    halden1 = g1.add_entity("Halden Energy")
    station1 = g1.add_entity("Vesterby Power Station")
    meridian1 = g1.add_entity("Meridian Rail")
    kalden1 = g1.add_entity("Kalden")
    closure1 = fact(g1, "close", agent=halden1, patients=(station1,))
    suspension1 = fact(g1, "suspend services to", agent=meridian1, patients=(kalden1,))
    cause1 = fact(g1, "cause", agent=closure1, patients=(suspension1,))

    g2 = Graph(id="article-2")
    halden2 = g2.add_entity("Halden Energy")
    station2 = g2.add_entity("Vesterby Power Station")
    meridian2 = g2.add_entity("Meridian Rail")
    kalden2 = g2.add_entity("Kalden")
    closure2 = fact(g2, "shut down", agent=halden2, patients=(station2,))
    suspension2 = fact(g2, "halt services to", agent=meridian2, patients=(kalden2,))
    cause2 = fact(g2, "result in", agent=closure2, patients=(suspension2,))

    _, groups, _ = match_graphs([g1, g2])

    assert {frozenset(group) for group in groups} == _groups(
        (halden1, halden2),
        (station1, station2),
        (meridian1, meridian2),
        (kalden1, kalden2),
        (closure1, closure2),
        (suspension1, suspension2),
        (cause1, cause2),
    )


def test_qualifier_selects_the_corresponding_event_occurrence():
    """Two purchase events with the same participants in one article: the
    qualified occurrence matches the qualified event in the other article,
    and the unqualified occurrence stays unmatched."""
    g1 = Graph(id="article-1")
    acme1 = g1.add_entity("Acme Corp")
    datavault1 = g1.add_entity("DataVault")
    price1 = g1.add_entity("$4 billion")
    purchase1 = fact(g1, "acquire", agent=acme1, patients=(datavault1, price1))

    g2 = Graph(id="article-2")
    acme2 = g2.add_entity("Acme Corp")
    datavault2 = g2.add_entity("DataVault")
    price2 = g2.add_entity("$4 billion")
    qualified_purchase2 = fact(g2, "purchase", agent=acme2, patients=(datavault2, price2))
    other_purchase2 = fact(g2, "purchase", agent=acme2, patients=(datavault2,))

    _, groups, unified = match_graphs([g1, g2])

    assert {frozenset(group) for group in groups} == _groups(
        (acme1, acme2),
        (datavault1, datavault2),
        (price1, price2),
        (purchase1, qualified_purchase2),
    )
    assert all(other_purchase2.id not in group for group in groups)
    _assert_original_occurrences(unified, g1, g2)


def test_matched_event_reinforces_name_variant_entity():
    """The core use case: a name variant ('Meridian Tech') merges with
    'Meridian Technologies' when the surrounding event structure matches."""
    g1 = Graph(id="article-1")
    buyer1 = g1.add_entity("Meridian Technologies")
    target1 = g1.add_entity("DataVault")
    acquisition = fact(g1, "acquire", agent=buyer1, patients=(target1,))

    g2 = Graph(id="article-2")
    buyer2 = g2.add_entity("Meridian Tech")
    target2 = g2.add_entity("DataVault")
    purchase = fact(g2, "purchase", agent=buyer2, patients=(target2,))

    _, groups, _ = match_graphs([g1, g2])

    assert {frozenset(group) for group in groups} == _groups(
        (buyer1, buyer2),
        (target1, target2),
        (acquisition, purchase),
    )
