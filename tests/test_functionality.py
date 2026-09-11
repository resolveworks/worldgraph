"""Unit tests for per-slot functionality.

Functionality is computed per slot (subject/object) over participation in
statements. Forward functionality — how discriminative an alignment
through the slot is, from the statement's side — is 1 by construction (a
statement has exactly one subject and one object). Inverse functionality
— how much a participant in the slot determines the statement — pools
entity participants by primary name across graphs, while statement
participants pool by occurrence (each statement is an instance).
"""

import pytest

from worldgraph.graph import Graph
from worldgraph.match import OBJECT, SUBJECT, compute_functionality


def test_forward_functionality_is_one_by_construction():
    """Every statement has exactly one occupant per slot, so an alignment
    through either slot is fully discriminative from the statement side."""
    g = Graph(id="g1")
    acme = g.add_entity("Acme Corp")
    beta = g.add_entity("Beta Systems")
    nord = g.add_entity("Nordwind")
    baltic = g.add_entity("Baltic Freight")
    g.add_statement(acme, "acquire", beta)
    g.add_statement(nord, "acquire", baltic)

    func = compute_functionality([g])
    assert func[SUBJECT].forward == 1.0
    assert func[OBJECT].forward == 1.0


def test_shared_subject_name_halves_inverse_functionality():
    """A participant name appearing as subject of two statements is only
    half-discriminative: aligning through it carries half the evidence."""
    g = Graph(id="g1")
    john = g.add_entity("John Mercer")
    acme = g.add_entity("Acme Corp")
    beta = g.add_entity("Beta Systems")
    g.add_statement(john, "join", acme)
    g.add_statement(john, "lead", beta)

    func = compute_functionality([g])
    assert func[SUBJECT].inverse == pytest.approx(0.5)
    assert func[OBJECT].inverse == pytest.approx(1.0)


def test_entity_participants_pool_by_name_across_graphs():
    """The same participant reported by two outlets pools its statistics:
    "John" as subject of one statement per graph halves inverse
    functionality, exactly as two same-graph statements would."""
    g1 = Graph(id="g1")
    john1 = g1.add_entity("John Mercer")
    g1.add_statement(john1, "join", g1.add_entity("Acme Corp"))

    g2 = Graph(id="g2")
    john2 = g2.add_entity("John Mercer")
    g2.add_statement(john2, "lead", g2.add_entity("Beta Systems"))

    func = compute_functionality([g1, g2])
    assert func[SUBJECT].inverse == pytest.approx(0.5)


def test_statement_participants_pool_by_occurrence():
    """Statements as participants are instances: two qualifier statements
    about two different statements never pool, while their identical-name
    title participant does."""
    g1 = Graph(id="g1")
    s1 = g1.add_statement(g1.add_entity("Acme Corp"), "acquire", g1.add_entity("Beta Systems"))
    ceo1 = g1.add_entity("CEO")
    g1.add_statement(s1, "as", ceo1)

    g2 = Graph(id="g2")
    s2 = g2.add_statement(g2.add_entity("Nordwind"), "acquire", g2.add_entity("Baltic Freight"))
    ceo2 = g2.add_entity("CEO")
    g2.add_statement(s2, "as", ceo2)

    func = compute_functionality([g1, g2])
    # Subjects: acme(1), nordwind(1), s1(1), s2(1) — occurrences never pool
    assert func[SUBJECT].inverse == pytest.approx(1.0)
    # Objects: beta(1), baltic(1), "CEO"(2 across both qualifier statements)
    assert func[OBJECT].inverse == pytest.approx(0.75)
