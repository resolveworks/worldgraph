"""Output tests: run_matching produces the merged canonical graph, and
the CLI match subcommand is wired to it."""

import json

from click.testing import CliRunner

from worldgraph.cli import cli
from worldgraph.graph import Entity, Graph, Statement, load_graph, save_graph
from worldgraph.match import run_matching


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

    run_matching([p1, p2], output)

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


def test_cli_match_writes_output(tmp_path):
    g1 = _deal_graph("g1", "acquire", ["Acme Corp"])
    g2 = _deal_graph("g2", "purchase", ["Acme Corp"])
    p1 = tmp_path / "g1.json"
    p2 = tmp_path / "g2.json"
    save_graph(g1, p1)
    save_graph(g2, p2)
    output = tmp_path / "merged.json"

    result = CliRunner().invoke(cli, ["match", str(p1), str(p2), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert len(load_graph(output).terms) == 3
