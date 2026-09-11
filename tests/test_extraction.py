"""Tests for the extraction schema: one id namespace across entities and
statements, reference resolvability, no direct self-reference, and
conversion to a runtime term graph. Forward references between
statements are valid; statements may be about statements."""

import pytest
from pydantic import ValidationError

from worldgraph.extract import (
    EntityRef,
    Extraction,
    StatementModel,
    extraction_to_graph,
)
from worldgraph.graph import Entity, Statement


def entities(*names: str) -> list[EntityRef]:
    return [EntityRef(id=f"e{i + 1}", name=name) for i, name in enumerate(names)]


def test_duplicate_entity_id_raises():
    with pytest.raises(ValidationError, match="duplicate term ids"):
        Extraction(
            entities=[EntityRef(id="e1", name="Alice"), EntityRef(id="e1", name="Bob")],
            statements=[],
        )


def test_duplicate_statement_id_raises():
    with pytest.raises(ValidationError, match="duplicate term ids"):
        Extraction(
            entities=entities("Alice", "Bob"),
            statements=[
                StatementModel(id="s1", subject="e1", predicate="know", object="e2"),
                StatementModel(id="s1", subject="e2", predicate="know", object="e1"),
            ],
        )


def test_entity_statement_id_collision_raises():
    """Entities and statements share one id namespace — a collision would
    make references ambiguous."""
    with pytest.raises(ValidationError, match="duplicate term ids"):
        Extraction(
            entities=entities("Alice"),
            statements=[
                StatementModel(id="e1", subject="e1", predicate="resign from", object="e1"),
            ],
        )


def test_unknown_reference_raises():
    """References that resolve to neither an entity nor a statement are
    rejected — never dropped or patched."""
    with pytest.raises(ValidationError, match="unknown ids"):
        Extraction(
            entities=entities("Alice"),
            statements=[
                StatementModel(id="s1", subject="e1", predicate="know", object="e99"),
            ],
        )


def test_statement_cannot_reference_itself():
    with pytest.raises(ValidationError, match="themselves"):
        Extraction(
            entities=entities("Alice"),
            statements=[
                StatementModel(id="s1", subject="s1", predicate="deny", object="e1"),
            ],
        )
    with pytest.raises(ValidationError, match="themselves"):
        Extraction(
            entities=entities("Alice"),
            statements=[
                StatementModel(id="s1", subject="e1", predicate="deny", object="s1"),
            ],
        )


def test_statement_about_statement_is_valid():
    """The design example: a qualifier is a statement about a statement."""
    ext = Extraction(
        entities=entities("Jane", "Supercorp", "CEO"),
        statements=[
            StatementModel(id="s1", subject="e1", predicate="work at", object="e2"),
            StatementModel(id="s2", subject="s1", predicate="as", object="e3"),
        ],
    )
    assert ext.statements[1].subject == "s1"


def test_forward_reference_is_valid():
    """A statement may be listed before the statement it references."""
    ext = Extraction(
        entities=entities("Ivo Brandt"),
        statements=[
            StatementModel(id="s2", subject="s1", predicate="join", object="e1"),
            StatementModel(id="s1", subject="e1", predicate="visit", object="e1"),
        ],
    )
    assert ext.statements[0].subject == "s1"


# ---------------------------------------------------------------------------
# Conversion to runtime graph
# ---------------------------------------------------------------------------


def jane_extraction() -> Extraction:
    return Extraction(
        entities=entities("Jane", "Supercorp", "CEO"),
        statements=[
            StatementModel(id="s1", subject="e1", predicate="work at", object="e2"),
            StatementModel(id="s2", subject="s1", predicate="as", object="e3"),
        ],
    )


def test_extraction_to_graph_structure():
    """Entities become entity terms, statements become statement terms,
    and nesting survives: the 'as' statement is about the 'work at'
    statement."""
    graph = extraction_to_graph("article-1", jane_extraction())

    entity_terms = [t for t in graph.terms.values() if isinstance(t, Entity)]
    statement_terms = [t for t in graph.terms.values() if isinstance(t, Statement)]
    assert len(entity_terms) == 3
    assert len(statement_terms) == 2

    as_statement = next(t for t in statement_terms if t.predicates == ["as"])
    work = graph.resolve(as_statement.subject)
    assert isinstance(work, Statement)
    assert work.predicates == ["work at"]
    jane = graph.resolve(work.subject)
    supercorp = graph.resolve(work.object)
    ceo = graph.resolve(as_statement.object)
    assert isinstance(jane, Entity)
    assert isinstance(supercorp, Entity)
    assert isinstance(ceo, Entity)
    assert jane.names == ["Jane"]
    assert supercorp.names == ["Supercorp"]
    assert ceo.names == ["CEO"]


def test_extraction_to_graph_provenance():
    """Every term carries the article graph id as provenance."""
    graph = extraction_to_graph("article-1", jane_extraction())

    assert graph.id == "article-1"
    assert all(term.graph_id == "article-1" for term in graph.terms.values())


def test_extraction_to_graph_order_independent():
    """Runtime ids are pre-allocated, so statements listed before the
    terms they reference still convert and validate."""
    ext = Extraction(
        entities=entities("Ivo Brandt"),
        statements=[
            StatementModel(id="s2", subject="s1", predicate="join", object="e1"),
            StatementModel(id="s1", subject="e1", predicate="visit", object="e1"),
        ],
    )
    graph = extraction_to_graph("article-1", ext)

    visit = next(t for t in graph.terms.values() if isinstance(t, Statement) and t.predicates == ["visit"])
    join = next(t for t in graph.terms.values() if isinstance(t, Statement) and t.predicates == ["join"])
    assert join.subject == visit.id


def test_extraction_to_graph_unique_runtime_ids():
    """Extraction-local ids ('e1', 's1') never collide at runtime — every
    term gets its own runtime id."""
    graph = extraction_to_graph("article-1", jane_extraction())

    assert len(graph.terms) == 5
    assert len(graph.terms) == len({term.id for term in graph.terms.values()})
