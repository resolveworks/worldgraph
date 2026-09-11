"""Shared term-graph data structures and I/O.

A graph is a set of **terms**: an entity (a named thing in the world) or
a statement (a fact the article asserts). A statement is a triple —
subject term, predicate phrase, object term — and since statements are
terms, a statement may be about a statement: qualifiers, attribution,
and claims-about-claims all take the same nested shape (RDF-star). The
direction of a fact lives in subject/object position only; there are no
roles, no node kinds, and no controlled vocabularies. Terms share one id
namespace per graph.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar


@dataclass
class Entity:
    id: str
    graph_id: str  # id of the article graph this term was extracted from
    names: list[str]


@dataclass
class Statement:
    """An asserted fact: ``subject`` and ``object`` reference term ids of
    this graph (either may be another statement), and ``predicate`` is a
    short verb phrase in active voice — the subject is the one who brings
    the fact about."""

    id: str
    graph_id: str  # id of the article graph this term was extracted from
    subject: str  # term id
    predicate: str
    object: str  # term id


Term = Entity | Statement

T = TypeVar("T", bound=Term)


@dataclass
class Graph:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    terms: dict[str, Term] = field(default_factory=dict)

    def add_entity(self, names: str | list[str], id: str | None = None) -> Entity:
        """Add an entity term with the given name(s)."""
        if isinstance(names, str):
            names = [names]
        entity = Entity(
            id=id if id is not None else str(uuid.uuid4()),
            graph_id=self.id,
            names=names,
        )
        return self._insert(entity)

    def add_statement(
        self,
        subject: Term | str,
        predicate: str,
        object: Term | str,
        id: str | None = None,
    ) -> Statement:
        """Add a statement term. Endpoints may be given as term objects or
        as ids. Ids (and the optional explicit term ``id``) exist for
        construction flexibility — forward references to terms not yet
        added are fine; ``validate()`` checks resolvability once
        construction is complete."""
        statement = Statement(
            id=id if id is not None else str(uuid.uuid4()),
            graph_id=self.id,
            subject=subject.id if isinstance(subject, (Entity, Statement)) else subject,
            predicate=predicate,
            object=object.id if isinstance(object, (Entity, Statement)) else object,
        )
        return self._insert(statement)

    def resolve(self, term_id: str) -> Term:
        """Return the term with the given id."""
        try:
            return self.terms[term_id]
        except KeyError:
            raise ValueError(f"unknown term id: {term_id!r}") from None

    def _insert(self, term: T) -> T:
        if term.id in self.terms:
            raise ValueError(f"duplicate term id: {term.id!r}")
        self.terms[term.id] = term
        return term

    def validate(self) -> None:
        """Check the structural invariants.

        Every statement endpoint must resolve to a term of this graph, and
        no statement may directly participate in itself. Cycles and
        forward references are valid.
        """
        for term in self.terms.values():
            if not isinstance(term, Statement):
                continue
            for endpoint in (term.subject, term.object):
                if endpoint not in self.terms:
                    raise ValueError(
                        f"statement {term.id!r} references unknown term id: {endpoint!r}"
                    )
                if endpoint == term.id:
                    raise ValueError(
                        f"statement participates in itself: {term.id!r}"
                    )


_ENTITY_FIELDS = frozenset({"type", "id", "graph_id", "names"})
_STATEMENT_FIELDS = frozenset({"type", "id", "graph_id", "subject", "predicate", "object"})


def _check_fields(term_data: dict, expected: frozenset[str]) -> None:
    unknown = sorted(set(term_data) - expected)
    if unknown:
        raise ValueError(
            f"unknown fields on {term_data.get('type')!r} term: {unknown}"
        )


def load_graph(path: Path) -> Graph:
    """Load a single graph JSON file.

    Raises on duplicate term ids, unresolvable statement references, and
    unknown fields — invalid state is never silently repaired.
    """
    with open(path) as f:
        data = json.load(f)

    unknown = sorted(set(data) - {"id", "terms", "matches"})
    if unknown:
        raise ValueError(f"unknown graph fields: {unknown}")

    terms: dict[str, Term] = {}
    for term_data in data["terms"]:
        term_type = term_data.get("type")
        if term_type == "entity":
            _check_fields(term_data, _ENTITY_FIELDS)
            term: Term = Entity(
                id=term_data["id"],
                graph_id=term_data["graph_id"],
                names=term_data["names"],
            )
        elif term_type == "statement":
            _check_fields(term_data, _STATEMENT_FIELDS)
            term = Statement(
                id=term_data["id"],
                graph_id=term_data["graph_id"],
                subject=term_data["subject"],
                predicate=term_data["predicate"],
                object=term_data["object"],
            )
        else:
            raise ValueError(f"unknown term type: {term_type!r}")
        if term.id in terms:
            raise ValueError(f"duplicate term id: {term.id!r}")
        terms[term.id] = term

    graph = Graph(id=data["id"], terms=terms)
    graph.validate()
    return graph


def save_graph(
    graph: Graph,
    path: Path,
    matches: list[list[str]] | None = None,
) -> None:
    """Write graph to JSON, with optional match groups. Validates first."""
    graph.validate()

    terms_out = []
    for term in graph.terms.values():
        if isinstance(term, Entity):
            terms_out.append(
                {
                    "type": "entity",
                    "id": term.id,
                    "graph_id": term.graph_id,
                    "names": term.names,
                }
            )
        else:
            terms_out.append(
                {
                    "type": "statement",
                    "id": term.id,
                    "graph_id": term.graph_id,
                    "subject": term.subject,
                    "predicate": term.predicate,
                    "object": term.object,
                }
            )

    output = {
        "id": graph.id,
        "terms": terms_out,
        "matches": matches or [],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
