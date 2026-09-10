"""Tests for the extraction schema: unique ids, valid references, and the
conversion to a runtime graph. Relation endpoints may reference relations,
including forward references."""

import pytest
from pydantic import ValidationError

from worldgraph.extract import Entity, Extraction, Relation, extraction_to_graph
from worldgraph.graph import Edge, Node


def entities(*names: str) -> list[Entity]:
    return [Entity(id=f"e{i + 1}", name=name) for i, name in enumerate(names)]


def test_duplicate_entity_id_raises():
    with pytest.raises(ValidationError, match="duplicate entity ids"):
        Extraction(
            entities=[Entity(id="e1", name="Alice"), Entity(id="e1", name="Bob")],
            relations=[],
        )


def test_duplicate_relation_id_raises():
    with pytest.raises(ValidationError, match="duplicate relation ids"):
        Extraction(
            entities=entities("Alice", "Bob"),
            relations=[
                Relation(id="r1", source="e1", target="e2",
                         relation="knows"),
                Relation(id="r1", source="e2", target="e1",
                         relation="knows"),
            ],
        )


def test_entity_relation_id_collision_raises():
    """Entities and relations share one reference namespace — a collision
    would make references ambiguous."""
    with pytest.raises(ValidationError, match="disjoint"):
        Extraction(
            entities=entities("Alice", "Bob"),
            relations=[
                Relation(id="e1", source="e1", target="e2",
                         relation="knows"),
            ],
        )


def test_unknown_reference_raises():
    """References that resolve to neither an entity nor a relation are
    rejected — never dropped or patched."""
    with pytest.raises(ValidationError, match="unknown ids"):
        Extraction(
            entities=entities("Alice", "Bob"),
            relations=[
                Relation(id="r1", source="e1", target="e99",
                         relation="knows"),
            ],
        )


# ---------------------------------------------------------------------------
# Conversion to runtime graph
# ---------------------------------------------------------------------------


def qualifier_extraction() -> Extraction:
    """The Corin example, with the qualifier listed before the relation it
    references: r2 (for) points at r1 (manage) before r1 is defined."""
    return Extraction(
        entities=entities("Tessa Corin", "Halden Freight", "Vesterby"),
        relations=[
            Relation(id="r2", source="r1", target="e3",
                     relation="for"),
            Relation(id="r1", source="e1", target="e2",
                     relation="manage"),
        ],
    )


def test_extraction_to_graph_preserves_references():
    """Relation→relation references point at the runtime edges of the
    referenced relations, including forward references."""
    graph = extraction_to_graph("article-1", qualifier_extraction())

    r1 = next(e for e in graph.edges.values() if e.relation == "manage")
    r2 = next(e for e in graph.edges.values() if e.relation == "for")

    assert r2.source == r1.id  # forward reference resolved
    assert isinstance(graph.resolve(r1.source), Node)
    assert isinstance(graph.resolve(r1.target), Node)
    assert isinstance(graph.resolve(r2.source), Edge)
    assert isinstance(graph.resolve(r2.target), Node)


def test_extraction_to_graph_provenance():
    """Every node and edge carries the article graph id as provenance."""
    graph = extraction_to_graph("article-1", qualifier_extraction())

    assert all(node.graph_id == "article-1" for node in graph.nodes.values())
    assert all(edge.graph_id == "article-1" for edge in graph.edges.values())


def test_extraction_to_graph_unique_term_ids():
    """Entities and relations map to distinct runtime ids — extraction-local
    ids ('e1', 'r1') never collide at runtime."""
    graph = extraction_to_graph("article-1", qualifier_extraction())

    term_ids = set(graph.nodes) | set(graph.edges)
    assert len(term_ids) == len(graph.nodes) + len(graph.edges)
