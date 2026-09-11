"""Tests for graph save/load round-trip and structural validation of the
term model: nesting, cycles, forward references, and the single id
namespace."""

import json
from pathlib import Path

import pytest

from worldgraph.graph import Entity, Graph, Statement, load_graph, save_graph


def jane_graph() -> Graph:
    """The design example: Jane works at Supercorp as CEO, with the title
    as a statement about the work statement."""
    g = Graph(id="article-1")
    jane = g.add_entity("Jane")
    supercorp = g.add_entity("Supercorp")
    ceo = g.add_entity("CEO")
    work = g.add_statement(jane, "work at", supercorp)
    g.add_statement(work, "as", ceo)
    return g


def test_roundtrip_preserves_structure_and_provenance(tmp_path: Path):
    """Terms, their references, and per-term provenance survive save/load."""
    g = jane_graph()
    path = tmp_path / "g.json"
    save_graph(g, path)

    loaded = load_graph(path)
    assert loaded.id == "article-1"
    assert loaded.terms == g.terms
    assert all(term.graph_id == "article-1" for term in loaded.terms.values())

    with open(path) as f:
        data = json.load(f)
    for term_data in data["terms"]:
        assert term_data["graph_id"] == "article-1"


def test_roundtrip_multi_name_entity(tmp_path: Path):
    """Entities with multiple names survive save/load round-trip."""
    g = Graph(id="article-1")
    meridian = g.add_entity(["Meridian Technologies", "Meridian Tech"])
    datavault = g.add_entity("DataVault")
    g.add_statement(meridian, "acquire", datavault)

    path = tmp_path / "g.json"
    save_graph(g, path)

    loaded = load_graph(path)
    loaded_meridian = loaded.terms[meridian.id]
    assert isinstance(loaded_meridian, Entity)
    assert loaded_meridian.names == ["Meridian Technologies", "Meridian Tech"]


def test_roundtrip_nested_statements(tmp_path: Path):
    """A statement about a statement (a qualifier) round-trips with its
    endpoints intact."""
    g = jane_graph()
    path = tmp_path / "g.json"
    save_graph(g, path)
    loaded = load_graph(path)

    as_statement = next(
        t
        for t in loaded.terms.values()
        if isinstance(t, Statement) and t.predicates == ["as"]
    )
    work = loaded.resolve(as_statement.subject)
    assert isinstance(work, Statement)
    assert work.predicates == ["work at"]
    ceo = loaded.resolve(as_statement.object)
    assert isinstance(ceo, Entity)
    assert ceo.names == ["CEO"]


def test_cycle_is_valid_and_roundtrips(tmp_path: Path):
    """Two statements about each other form a cycle — valid, and the
    references survive save/load."""
    g = Graph(id="article-1")
    a = g.add_entity("A")
    g.add_statement(a, "say", "t2", id="t1")
    g.add_statement("t1", "contradict", a, id="t2")
    g.validate()

    path = tmp_path / "g.json"
    save_graph(g, path)
    loaded = load_graph(path)

    s1 = loaded.resolve("t1")
    s2 = loaded.resolve("t2")
    assert isinstance(s1, Statement)
    assert isinstance(s2, Statement)
    assert s1.object == "t2"
    assert s2.subject == "t1"


def test_forward_reference_is_valid():
    """A statement may reference a term id added later — validation is
    deferred until construction is complete."""
    g = Graph(id="article-1")
    jane = g.add_entity("Jane")
    g.add_statement(jane, "work at", "e-supercorp")
    g.add_entity("Supercorp", id="e-supercorp")
    g.validate()  # does not raise


def test_add_statement_accepts_terms_and_ids():
    """Endpoints may be term objects or raw ids; both yield the same
    stored references."""
    g = Graph(id="article-1")
    a = g.add_entity("A")
    by_object = g.add_statement(a, "know", "b-id")
    b = g.add_entity("B", id="b-id")
    by_object_and_term = g.add_statement(a, "know", b)

    assert by_object.object == "b-id"
    assert by_object_and_term.object == b.id


def test_resolve_returns_term_and_raises_on_unknown():
    g = jane_graph()
    work = next(
        t for t in g.terms.values() if isinstance(t, Statement)
    )
    jane = g.resolve(work.subject)
    assert isinstance(jane, Entity)
    assert jane.names == ["Jane"]

    with pytest.raises(ValueError, match="unknown term id"):
        g.resolve("no-such-term")


def test_duplicate_term_id_raises():
    """Terms share one id namespace — an entity and a statement cannot
    take the same id, nor can two entities."""
    g = Graph(id="article-1")
    g.add_entity("A", id="t1")
    with pytest.raises(ValueError, match="duplicate term id"):
        g.add_entity("B", id="t1")
    with pytest.raises(ValueError, match="duplicate term id"):
        g.add_statement("t1", "know", "t1", id="t1")


def test_validate_unknown_reference_raises():
    """Statement endpoints must resolve to terms of this graph — invalid
    references are rejected, never dropped or repaired."""
    g = Graph(id="article-1")
    a = g.add_entity("A")
    g.add_statement(a, "know", "ghost")

    with pytest.raises(ValueError, match="unknown term id"):
        g.validate()


def test_validate_direct_self_participation_raises():
    """A statement cannot be its own subject or object; indirect cycles
    remain valid."""
    g = Graph(id="article-1")
    a = g.add_entity("A")
    g.add_statement(a, "know", "t1", id="t1")
    with pytest.raises(ValueError, match="participates in itself"):
        g.validate()

    g2 = Graph(id="article-1")
    a2 = g2.add_entity("A")
    g2.add_statement("t1", "know", a2, id="t1")
    with pytest.raises(ValueError, match="participates in itself"):
        g2.validate()


def test_save_validates_first(tmp_path: Path):
    """save_graph refuses to write an invalid graph."""
    g = Graph(id="article-1")
    a = g.add_entity("A")
    g.add_statement(a, "know", "ghost")

    with pytest.raises(ValueError, match="unknown term id"):
        save_graph(g, tmp_path / "g.json")
    assert not (tmp_path / "g.json").exists()


def test_load_duplicate_term_id_raises(tmp_path: Path):
    """Duplicate term ids are invalid — no silent overwrite."""
    data = {
        "id": "article-1",
        "terms": [
            {"type": "entity", "id": "t1", "graph_id": "article-1", "names": ["Alice"]},
            {"type": "entity", "id": "t1", "graph_id": "article-1", "names": ["Bob"]},
        ],
        "matches": [],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="duplicate term id"):
        load_graph(path)


def test_load_unknown_reference_raises(tmp_path: Path):
    data = {
        "id": "article-1",
        "terms": [
            {
                "type": "statement",
                "id": "s1",
                "graph_id": "article-1",
                "subject": "t999",
                "predicates": ["know"],
                "object": "t998",
            },
        ],
        "matches": [],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unknown term id"):
        load_graph(path)


def test_load_unknown_field_raises(tmp_path: Path):
    """Unknown fields — top-level or on a term — are invalid state."""
    base = {
        "id": "article-1",
        "terms": [
            {
                "type": "entity",
                "id": "t1",
                "graph_id": "article-1",
                "names": ["Alice"],
                "role": "agent",
            },
        ],
        "matches": [],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="unknown fields"):
        load_graph(path)

    top_level = {"id": "article-1", "terms": [], "matches": [], "edges": []}
    path.write_text(json.dumps(top_level))
    with pytest.raises(ValueError, match="unknown graph fields"):
        load_graph(path)


def test_load_unknown_term_type_raises(tmp_path: Path):
    data = {
        "id": "article-1",
        "terms": [{"type": "event", "id": "t1", "graph_id": "article-1", "names": ["x"]}],
        "matches": [],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unknown term type"):
        load_graph(path)


def test_load_missing_field_raises(tmp_path: Path):
    """A term lacking a required field is invalid — no fallback."""
    data = {
        "id": "article-1",
        "terms": [
            {"type": "statement", "id": "s1", "graph_id": "article-1", "predicates": ["know"]},
        ],
        "matches": [],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    with pytest.raises(KeyError):
        load_graph(path)


def test_matches_roundtrip(tmp_path: Path):
    """Optional match groups are written through to the output file."""
    g = jane_graph()
    path = tmp_path / "g.json"
    jane = next(t.id for t in g.terms.values() if isinstance(t, Entity))
    save_graph(g, path, matches=[[jane]])

    with open(path) as f:
        data = json.load(f)
    assert data["matches"] == [[jane]]
