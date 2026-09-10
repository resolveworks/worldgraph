"""Public-interface tests for matching nodes and edges as graph terms."""

import pytest

from worldgraph.graph import Graph
from worldgraph.match import match_graphs


def _group_containing(groups: list[set[str]], term_id: str) -> set[str] | None:
    return next((group for group in groups if term_id in group), None)


def _assert_grouped_pair(groups: list[set[str]], left: str, right: str) -> None:
    assert _group_containing(groups, left) == {left, right}


def test_identical_fact_matches_both_nodes_and_edge(embedder):
    first = Graph(id="article-1")
    acme_1 = first.add_entity("Acme Corp")
    gamma_1 = first.add_entity("Gamma AI")
    acquisition_1 = first.add_edge(acme_1, gamma_1, "acquired", "past")

    second = Graph(id="article-2")
    acme_2 = second.add_entity("Acme Corp")
    gamma_2 = second.add_entity("Gamma AI")
    acquisition_2 = second.add_edge(acme_2, gamma_2, "acquired", "past")

    _, groups, unified = match_graphs([first, second], embedder)

    _assert_grouped_pair(groups, acme_1.id, acme_2.id)
    _assert_grouped_pair(groups, gamma_1.id, gamma_2.id)
    _assert_grouped_pair(groups, acquisition_1.id, acquisition_2.id)
    assert set(unified.nodes) == {acme_1.id, gamma_1.id, acme_2.id, gamma_2.id}
    assert set(unified.edges) == {acquisition_1.id, acquisition_2.id}


def test_synonymous_relation_names_match_when_endpoints_match(embedder):
    first = Graph(id="article-1")
    buyer_1 = first.add_entity("Acme Corp")
    target_1 = first.add_entity("Gamma AI")
    acquisition = first.add_edge(buyer_1, target_1, "acquired", "past")

    second = Graph(id="article-2")
    buyer_2 = second.add_entity("Acme Corp")
    target_2 = second.add_entity("Gamma AI")
    takeover = second.add_edge(buyer_2, target_2, "took control of", "past")

    _, groups, _ = match_graphs([first, second], embedder)

    _assert_grouped_pair(groups, buyer_1.id, buyer_2.id)
    _assert_grouped_pair(groups, target_1.id, target_2.id)
    _assert_grouped_pair(groups, acquisition.id, takeover.id)


@pytest.mark.parametrize("endpoint_arrangement", ["swapped", "unrelated"])
def test_same_relation_name_does_not_match_with_wrong_endpoints(
    embedder, endpoint_arrangement
):
    first = Graph(id="article-1")
    acme_1 = first.add_entity("Acme Corp")
    gamma_1 = first.add_entity("Gamma AI")
    acquisition_1 = first.add_edge(acme_1, gamma_1, "acquired", "past")

    second = Graph(id="article-2")
    if endpoint_arrangement == "swapped":
        source = second.add_entity("Gamma AI")
        target = second.add_entity("Acme Corp")
    else:
        source = second.add_entity("Northstar Labs")
        target = second.add_entity("CloudScale")
    acquisition_2 = second.add_edge(source, target, "acquired", "past")

    _, groups, _ = match_graphs([first, second], embedder)

    assert not any({acquisition_1.id, acquisition_2.id} <= group for group in groups)


def test_matched_edge_structure_reinforces_name_variant_node(embedder):
    first = Graph(id="article-1")
    buyer_1 = first.add_entity("Meridian Technologies")
    target_1 = first.add_entity("DataVault")
    acquisition_1 = first.add_edge(buyer_1, target_1, "acquired", "past")

    second = Graph(id="article-2")
    buyer_2 = second.add_entity("Meridian Tech")
    target_2 = second.add_entity("DataVault")
    acquisition_2 = second.add_edge(buyer_2, target_2, "acquired", "past")

    _, groups, _ = match_graphs([first, second], embedder)

    _assert_grouped_pair(groups, target_1.id, target_2.id)
    _assert_grouped_pair(groups, acquisition_1.id, acquisition_2.id)
    _assert_grouped_pair(groups, buyer_1.id, buyer_2.id)


def test_temporal_disagreement_does_not_veto_edge_match(embedder):
    """Matching endpoints and predicate can outweigh temporal disagreement.

    The differing temporal values provide no agreement, but remain source-local
    observations on the two matched edge occurrences in the unified graph.
    """
    first = Graph(id="completion")
    acme_1 = first.add_entity("Acme Corp")
    gamma_1 = first.add_entity("Gamma AI")
    completed = first.add_edge(acme_1, gamma_1, "acquire", "past")

    second = Graph(id="announcement")
    acme_2 = second.add_entity("Acme Corp")
    gamma_2 = second.add_entity("Gamma AI")
    announced = second.add_edge(acme_2, gamma_2, "acquire", "future")

    _, groups, unified = match_graphs([first, second], embedder)

    _assert_grouped_pair(groups, acme_1.id, acme_2.id)
    _assert_grouped_pair(groups, gamma_1.id, gamma_2.id)
    _assert_grouped_pair(groups, completed.id, announced.id)
    assert unified.edges[completed.id] == completed
    assert unified.edges[announced.id] == announced
    assert {
        unified.edges[edge_id].temporal for edge_id in (completed.id, announced.id)
    } == {
        "past",
        "future",
    }
