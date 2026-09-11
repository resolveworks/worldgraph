"""Output tests: run_matching produces the merged canonical graph,
with the predicate prior built from an injected stub embedder."""

import json

import numpy as np

from worldgraph.embed import Embedder
from worldgraph.graph import Entity, Graph, Statement, load_graph, save_graph
from worldgraph.match import run_matching


class StubEmbedder(Embedder):
    """Returns an explicit predicate → vector dict; never a real model."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = vectors

    def embed(self, keys, template=None):
        return {k: self._vectors[k] for k in keys}


def _deal_graph(graph_id: str, predicate: str, acme_names: list[str]) -> Graph:
    g = Graph(id=graph_id)
    g.add_entity(acme_names, id="e1")
    g.add_entity("Beta Systems", id="e2")
    g.add_statement("e1", predicate, "e2", id="s1")
    return g


def test_run_matching_writes_merged_graph_and_matches(tmp_path):
    """The merged graph keeps one canonical term per group — entities
    carry the union of their name lists as aliases, statements appear
    once with endpoints remapped to canonical ids — and the matches field
    lists the groups as qualified original ids."""
    g1 = _deal_graph("g1", "acquire", ["Acme Corp", "Acme"])
    g2 = _deal_graph("g2", "purchase", ["Acme Corp"])
    p1 = tmp_path / "g1.json"
    p2 = tmp_path / "g2.json"
    save_graph(g1, p1)
    save_graph(g2, p2)
    output = tmp_path / "merged.json"

    stub = StubEmbedder(
        {
            "acquire": np.array([1.0, 0.0]),
            "purchase": np.array([0.97, 0.243]),
        }
    )
    run_matching([p1, p2], output, embedder=stub)

    merged = load_graph(output)
    assert set(merged.terms) == {"g1:e1", "g1:e2", "g1:s1"}

    acme = merged.resolve("g1:e1")
    assert isinstance(acme, Entity)
    assert acme.names == ["Acme Corp", "Acme"]

    statement = merged.resolve("g1:s1")
    assert isinstance(statement, Statement)
    assert statement.subject == "g1:e1"
    assert statement.object == "g1:e2"
    assert statement.predicate == "acquire"

    with open(output) as f:
        data = json.load(f)
    assert data["matches"] == [
        ["g1:e1", "g2:e1"],
        ["g1:e2", "g2:e2"],
        ["g1:s1", "g2:s1"],
    ]
