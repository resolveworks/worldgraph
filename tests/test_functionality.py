"""Unit tests for compute_functionality.

Functionality is computed per role over participation edges. Sources are
event nodes, counted per occurrence — events are instances, so identical
labels never pool. Targets are counted by name, so the same entity
mentioned across graphs pools its statistics.
"""

import pytest
from conftest import fact

from worldgraph.graph import Graph
from worldgraph.match import compute_functionality


def test_one_participant_per_role_functionality_is_1():
    """Events mapping to exactly one participant per role → functionality 1."""
    g = Graph(id="g1")
    acme = g.add_entity("Acme Corp")
    gamma = g.add_entity("Gamma AI")
    nord = g.add_entity("Nordwind")
    baltic = g.add_entity("Baltic Freight")
    fact(g, "acquire", agent=acme, patient=gamma)
    fact(g, "acquire", agent=nord, patient=baltic)

    func = compute_functionality([g])
    assert func["agent"].forward == pytest.approx(1.0)
    assert func["patient"].forward == pytest.approx(1.0)


def test_qualifiers_on_their_own_roles_keep_functionality_at_1():
    """A qualified event ('manage' as managing director for Vesterby) has
    one participant per role: the title and scope party pool onto
    capacity and beneficiary instead of splitting patient, so every
    touched role keeps forward functionality 1."""
    g = Graph(id="g1")
    tessa = g.add_entity("Tessa Corin")
    halden = g.add_entity("Halden Freight")
    director = g.add_entity("managing director")
    vesterby = g.add_entity("Vesterby")
    fact(g, "manage", agent=tessa, patient=halden, capacity=director, beneficiary=vesterby)

    func = compute_functionality([g])
    assert func["agent"].forward == pytest.approx(1.0)
    assert func["patient"].forward == pytest.approx(1.0)
    assert func["capacity"].forward == pytest.approx(1.0)
    assert func["beneficiary"].forward == pytest.approx(1.0)


def test_price_gets_its_own_role_not_a_second_patient():
    """A takeover of DataVault for $4 billion: the monetary amount is a
    price participant, so patient keeps exactly one target per event →
    forward functionality 1 for every touched role."""
    g = Graph(id="g1")
    acme = g.add_entity("Acme Corp")
    datavault = g.add_entity("DataVault")
    price = g.add_entity("$4 billion")
    fact(g, "acquire", agent=acme, patient=datavault, price=price)

    func = compute_functionality([g])
    assert func["patient"].forward == pytest.approx(1.0)
    assert func["price"].forward == pytest.approx(1.0)
    assert func["agent"].forward == pytest.approx(1.0)


def test_multiple_patients_lower_forward_functionality():
    """One event with two genuine patients — an acquirer acquiring two
    distinct companies (coordinated objects) — has patient out-degree
    2 → 0.5."""
    g = Graph(id="g1")
    acme = g.add_entity("Acme Corp")
    datavault = g.add_entity("DataVault")
    shazam = g.add_entity("Shazam")
    fact(g, "acquire", agent=acme, patient=(datavault, shazam))

    func = compute_functionality([g])
    assert func["patient"].forward == pytest.approx(0.5)
    assert func["agent"].forward == pytest.approx(1.0)


def test_entity_agent_of_many_events_lowers_inverse_functionality():
    """An entity that is the agent of two events has in-degree 2 via 'agent'."""
    g = Graph(id="g1")
    john = g.add_entity("John")
    mary = g.add_entity("Mary")
    car = g.add_entity("car")
    house = g.add_entity("house")
    fact(g, "purchase", agent=john, patient=car)
    fact(g, "purchase", agent=john, patient=house)
    fact(g, "resign", agent=mary)

    func = compute_functionality([g])
    # in-degrees via agent: John 2, Mary 1 → avg 1.5
    assert func["agent"].inverse == pytest.approx(2 / 3)


def test_target_names_pool_across_graphs():
    """The same target name in two graphs counts both event sources — a
    name participating in many events is a weak identity signal."""
    g1 = Graph(id="g1")
    acme1 = g1.add_entity("Acme Corp")
    gamma1 = g1.add_entity("Gamma AI")
    fact(g1, "acquire", agent=acme1, patient=gamma1)

    g2 = Graph(id="g2")
    acme2 = g2.add_entity("Acme Corp")
    gamma2 = g2.add_entity("Gamma AI")
    fact(g2, "purchase", agent=acme2, patient=gamma2)

    func = compute_functionality([g1, g2])
    # "Acme Corp" is agent of 2 event occurrences → in-degree 2 → 0.5
    assert func["agent"].inverse == pytest.approx(0.5)
    assert func["patient"].inverse == pytest.approx(0.5)


def test_event_sources_are_counted_per_occurrence():
    """Identically labeled events never pool: each occurrence has its own
    agent, so forward functionality stays 1 regardless of label overlap."""
    g1 = Graph(id="g1")
    acme = g1.add_entity("Acme Corp")
    gamma = g1.add_entity("Gamma AI")
    fact(g1, "acquire", agent=acme, patient=gamma)

    g2 = Graph(id="g2")
    nord = g2.add_entity("Nordwind")
    baltic = g2.add_entity("Baltic Freight")
    fact(g2, "acquire", agent=nord, patient=baltic)

    func = compute_functionality([g1, g2])
    assert func["agent"].forward == pytest.approx(1.0)
