"""Layer 3 integration tests.

These tests exercise the full matching pipeline: multiple graphs →
match_graphs.  They verify end-to-end correctness on multi-source
scenarios that L2 tests don't cover:

- Transitive merging across 3+ sources via union-find
- Cross-cluster isolation (independent stories don't merge, even with
  identical entity names or isomorphic structure)
- Cross-story entity linking (shared entity across clusters)
- Progressive merging does not cause cascading false merges
"""

from conftest import fact

from worldgraph.graph import Graph
from worldgraph.match import match_graphs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_group_containing(groups: list[set[str]], entity_id: str) -> set[str] | None:
    """Find the match group containing a given entity ID."""
    for group in groups:
        if entity_id in group:
            return group
    return None


# ---------------------------------------------------------------------------
# 1. Multi-source transitive merging
# ---------------------------------------------------------------------------


def test_three_source_with_person_name_variation():
    """Three sources with person name variations, each with enough
    structural context for propagation to work.

    "Dr. Priya Sharma" / "Priya Sharma" / "Dr. Sharma" — name similarity
    alone (~0.79) is below the merge threshold. Shared matched events
    (identical-name participants on both sides) provide the structural
    evidence to bridge the gap."""
    g1 = Graph(id="article-1")
    m1 = g1.add_entity("Meridian Technologies")
    p1 = g1.add_entity("Dr. Priya Sharma")
    j1 = g1.add_entity("James Chen")
    su1 = g1.add_entity("Stanford University")
    dv1 = g1.add_entity("DataVault Inc")
    nat1 = g1.add_entity("Nature")
    lab1 = g1.add_entity("Stanford AI Lab")
    fact(g1, "hire", agent=m1, patients=(p1,))
    fact(g1, "collaborate with", agent=p1, patients=(j1,))
    fact(g1, "be alumna of", agent=p1, patients=(su1,))
    fact(g1, "publish in", agent=p1, patients=(nat1,))
    fact(g1, "lead", agent=p1, patients=(lab1,))
    fact(g1, "acquire", agent=m1, patients=(dv1,))

    g2 = Graph(id="article-2")
    m2 = g2.add_entity("Meridian Technologies")
    p2 = g2.add_entity("Priya Sharma")
    j2 = g2.add_entity("James Chen")
    su2 = g2.add_entity("Stanford University")
    dv2 = g2.add_entity("DataVault Inc")
    nat2 = g2.add_entity("Nature")
    lab2 = g2.add_entity("Stanford AI Lab")
    fact(g2, "hire", agent=m2, patients=(p2,))
    fact(g2, "collaborate with", agent=p2, patients=(j2,))
    fact(g2, "be alumna of", agent=p2, patients=(su2,))
    fact(g2, "publish in", agent=p2, patients=(nat2,))
    fact(g2, "lead", agent=p2, patients=(lab2,))
    fact(g2, "acquire", agent=m2, patients=(dv2,))

    g3 = Graph(id="article-3")
    m3 = g3.add_entity("Meridian Technologies")
    p3 = g3.add_entity("Dr. Sharma")
    j3 = g3.add_entity("James Chen")
    su3 = g3.add_entity("Stanford University")
    dv3 = g3.add_entity("DataVault Inc")
    nat3 = g3.add_entity("Nature")
    lab3 = g3.add_entity("Stanford AI Lab")
    fact(g3, "hire", agent=m3, patients=(p3,))
    fact(g3, "collaborate with", agent=p3, patients=(j3,))
    fact(g3, "be alumna of", agent=p3, patients=(su3,))
    fact(g3, "publish in", agent=p3, patients=(nat3,))
    fact(g3, "lead", agent=p3, patients=(lab3,))
    fact(g3, "acquire", agent=m3, patients=(dv3,))

    graphs = [g1, g2, g3]
    _, groups, _ = match_graphs(graphs)

    m_group = _find_group_containing(groups, m1.id)
    assert m_group is not None, "Meridian entities not merged"
    assert m2.id in m_group and m3.id in m_group

    # p1-p2 and p1-p3 should each merge (name sim ~0.79 + structural
    # evidence from shared matched events). p2-p3 may not merge directly
    # (name sim ~0.39) but union-find transitivity through p1 links all three.
    p_group = _find_group_containing(groups, p1.id)
    assert p_group is not None, "Sharma entities not merged"
    assert p2.id in p_group and p3.id in p_group


# ---------------------------------------------------------------------------
# 2. Cross-cluster isolation
# ---------------------------------------------------------------------------


def test_identical_names_different_contexts_no_merge():
    """Two different people with identical names in unrelated stories.

    Story A: Dr. James Chen leads Advanced AI Lab, funded by NSF
    Story B: Dr. James Chen leads Climate Research Lab, funded by EPA

    Name similarity is 1.0 and both stories have the same event shape.
    But the participating entities are completely different — the event
    pairs suppress each other, and the negative evidence prevents merging
    the two James Chens."""
    # Story A: AI research
    a1 = Graph(id="ai-1")
    jc_a1 = a1.add_entity("Dr. James Chen")
    lab_a1 = a1.add_entity("Advanced AI Lab")
    nsf_a1 = a1.add_entity("National Science Foundation")
    fact(a1, "lead", agent=jc_a1, patients=(lab_a1,))
    fact(a1, "be funded by", agent=lab_a1, patients=(nsf_a1,))

    a2 = Graph(id="ai-2")
    jc_a2 = a2.add_entity("Dr. James Chen")
    lab_a2 = a2.add_entity("Advanced AI Lab")
    nsf_a2 = a2.add_entity("National Science Foundation")
    fact(a2, "head", agent=jc_a2, patients=(lab_a2,))
    fact(a2, "be funded by", agent=lab_a2, patients=(nsf_a2,))

    # Story B: climate research — same name, same structure, different entities
    b1 = Graph(id="climate-1")
    jc_b1 = b1.add_entity("Dr. James Chen")
    lab_b1 = b1.add_entity("Climate Research Lab")
    epa_b1 = b1.add_entity("Environmental Protection Agency")
    fact(b1, "lead", agent=jc_b1, patients=(lab_b1,))
    fact(b1, "be funded by", agent=lab_b1, patients=(epa_b1,))

    b2 = Graph(id="climate-2")
    jc_b2 = b2.add_entity("Dr. James Chen")
    lab_b2 = b2.add_entity("Climate Research Lab")
    epa_b2 = b2.add_entity("Environmental Protection Agency")
    fact(b2, "head", agent=jc_b2, patients=(lab_b2,))
    fact(b2, "be funded by", agent=lab_b2, patients=(epa_b2,))

    graphs = [a1, a2, b1, b2]
    _, groups, _ = match_graphs(graphs)

    cluster_a_ids = {jc_a1.id, lab_a1.id, nsf_a1.id, jc_a2.id, lab_a2.id, nsf_a2.id}
    cluster_b_ids = {jc_b1.id, lab_b1.id, epa_b1.id, jc_b2.id, lab_b2.id, epa_b2.id}

    for group in groups:
        has_a = bool(group & cluster_a_ids)
        has_b = bool(group & cluster_b_ids)
        assert not (has_a and has_b), (
            f"Cross-cluster merge between two different James Chens: {group}"
        )

    # Within-cluster merges should still work
    jc_a_group = _find_group_containing(groups, jc_a1.id)
    assert jc_a_group is not None and jc_a2.id in jc_a_group
    jc_b_group = _find_group_containing(groups, jc_b1.id)
    assert jc_b_group is not None and jc_b2.id in jc_b_group


# ---------------------------------------------------------------------------
# 3. Cross-story entity linking
# ---------------------------------------------------------------------------


def test_shared_entity_across_clusters():
    """An entity appearing in two independent stories should be linked
    across them, while story-specific entities stay isolated.

    Story A: Meridian Technologies acquired DataVault (CEO: Elena Vasquez)
    Story B: Meridian Technologies settles FTC investigation (CEO: Elena Vasquez)

    Meridian and Elena Vasquez are shared with identical names and matched
    CEO events across both stories. DataVault and FTC should NOT merge."""
    # Story A: acquisition (2 sources)
    a1 = Graph(id="acq-1")
    m_a1 = a1.add_entity("Meridian Technologies")
    dv_a1 = a1.add_entity("DataVault")
    ev_a1 = a1.add_entity("Elena Vasquez")
    fact(a1, "acquire", agent=m_a1, patients=(dv_a1,))
    fact(a1, "employ as CEO", agent=m_a1, patients=(ev_a1,))

    a2 = Graph(id="acq-2")
    m_a2 = a2.add_entity("Meridian Technologies")
    dv_a2 = a2.add_entity("DataVault")
    ev_a2 = a2.add_entity("Elena Vasquez")
    fact(a2, "purchase", agent=m_a2, patients=(dv_a2,))
    fact(a2, "employ as CEO", agent=m_a2, patients=(ev_a2,))

    # Story B: FTC investigation (2 sources) — shares Meridian + Elena
    b1 = Graph(id="ftc-1")
    m_b1 = b1.add_entity("Meridian Technologies")
    ftc_b1 = b1.add_entity("Federal Trade Commission")
    ev_b1 = b1.add_entity("Elena Vasquez")
    fact(b1, "investigate", agent=ftc_b1, patients=(m_b1,))
    fact(b1, "employ as CEO", agent=m_b1, patients=(ev_b1,))

    b2 = Graph(id="ftc-2")
    m_b2 = b2.add_entity("Meridian Technologies")
    ftc_b2 = b2.add_entity("Federal Trade Commission")
    ev_b2 = b2.add_entity("Elena Vasquez")
    fact(b2, "investigate", agent=ftc_b2, patients=(m_b2,))
    fact(b2, "employ as CEO", agent=m_b2, patients=(ev_b2,))

    graphs = [a1, a2, b1, b2]
    _, groups, _ = match_graphs(graphs)

    # All four Meridian entities should be in one group
    m_group = _find_group_containing(groups, m_a1.id)
    assert m_group is not None, "Meridian entities not merged"
    assert {m_a1.id, m_a2.id, m_b1.id, m_b2.id} <= m_group, (
        f"Not all Meridian entities merged across stories: {m_group}"
    )

    # DataVault should NOT merge with FTC
    dv_ids = {dv_a1.id, dv_a2.id}
    ftc_ids = {ftc_b1.id, ftc_b2.id}
    for group in groups:
        assert not (group & dv_ids and group & ftc_ids), (
            f"DataVault and FTC incorrectly merged: {group}"
        )


def test_shared_person_across_clusters():
    """A person entity shared across two stories, linked by identical
    names AND matched alumna events (Stanford University).

    Story A: Elena Vasquez is CEO of Meridian Technologies, alumna of Stanford
    Story B: Elena Vasquez keynotes Global Tech Summit, alumna of Stanford

    Meridian and Summit should NOT merge."""
    a1 = Graph(id="hire-1")
    m1 = a1.add_entity("Meridian Technologies")
    ev1 = a1.add_entity("Elena Vasquez")
    dv1 = a1.add_entity("DataVault Inc")
    su1 = a1.add_entity("Stanford University")
    fact(a1, "employ as CEO", agent=m1, patients=(ev1,))
    fact(a1, "acquire", agent=m1, patients=(dv1,))
    fact(a1, "be alumna of", agent=ev1, patients=(su1,))

    a2 = Graph(id="hire-2")
    m2 = a2.add_entity("Meridian Technologies")
    ev2 = a2.add_entity("Elena Vasquez")
    dv2 = a2.add_entity("DataVault Inc")
    su2 = a2.add_entity("Stanford University")
    fact(a2, "employ as CEO", agent=m2, patients=(ev2,))
    fact(a2, "acquire", agent=m2, patients=(dv2,))
    fact(a2, "be alumna of", agent=ev2, patients=(su2,))

    b1 = Graph(id="summit-1")
    ev3 = b1.add_entity("Elena Vasquez")
    summit1 = b1.add_entity("Global Tech Summit")
    su3 = b1.add_entity("Stanford University")
    fact(b1, "keynote", agent=ev3, patients=(summit1,))
    fact(b1, "be alumna of", agent=ev3, patients=(su3,))

    b2 = Graph(id="summit-2")
    ev4 = b2.add_entity("Elena Vasquez")
    summit2 = b2.add_entity("Global Tech Summit")
    su4 = b2.add_entity("Stanford University")
    fact(b2, "keynote", agent=ev4, patients=(summit2,))
    fact(b2, "be alumna of", agent=ev4, patients=(su4,))

    graphs = [a1, a2, b1, b2]
    _, groups, _ = match_graphs(graphs)

    # All four Elena Vasquez entities should merge (within + across stories)
    ev_group = _find_group_containing(groups, ev1.id)
    assert ev_group is not None, "Elena Vasquez entities not merged"
    assert {ev1.id, ev2.id, ev3.id, ev4.id} <= ev_group, (
        "Elena Vasquez not linked across stories"
    )

    # Meridian and Summit should NOT merge
    m_ids = {m1.id, m2.id}
    summit_ids = {summit1.id, summit2.id}
    for group in groups:
        assert not (group & m_ids and group & summit_ids), (
            f"Meridian and Summit incorrectly merged: {group}"
        )


# ---------------------------------------------------------------------------
# 4. Progressive merging does not cause cascading false merges
# ---------------------------------------------------------------------------


def test_progressive_merging_no_cascading_false_merges():
    """Two unrelated stories with isomorphic structure should stay
    separate even with progressive merging enabled.

    Story A: NovaTech acquired DataVault (CEO: James Chen)
    Story B: Quantum Labs acquired ClearSky (CEO: Sarah Park)

    Within-story entities merge across sources (identical names), but
    cross-story entities must not merge even after progressive merging
    enriches neighborhoods."""
    a1 = Graph(id="nova-1")
    nt_a1 = a1.add_entity("NovaTech")
    dv_a1 = a1.add_entity("DataVault")
    jc_a1 = a1.add_entity("James Chen")
    fact(a1, "acquire", agent=nt_a1, patients=(dv_a1,))
    fact(a1, "employ as CEO", agent=nt_a1, patients=(jc_a1,))

    a2 = Graph(id="nova-2")
    nt_a2 = a2.add_entity("NovaTech")
    dv_a2 = a2.add_entity("DataVault")
    jc_a2 = a2.add_entity("James Chen")
    fact(a2, "purchase", agent=nt_a2, patients=(dv_a2,))
    fact(a2, "employ as CEO", agent=nt_a2, patients=(jc_a2,))

    b1 = Graph(id="quantum-1")
    ql_b1 = b1.add_entity("Quantum Labs")
    cs_b1 = b1.add_entity("ClearSky")
    sp_b1 = b1.add_entity("Sarah Park")
    fact(b1, "acquire", agent=ql_b1, patients=(cs_b1,))
    fact(b1, "employ as CEO", agent=ql_b1, patients=(sp_b1,))

    b2 = Graph(id="quantum-2")
    ql_b2 = b2.add_entity("Quantum Labs")
    cs_b2 = b2.add_entity("ClearSky")
    sp_b2 = b2.add_entity("Sarah Park")
    fact(b2, "purchase", agent=ql_b2, patients=(cs_b2,))
    fact(b2, "employ as CEO", agent=ql_b2, patients=(sp_b2,))

    graphs = [a1, a2, b1, b2]
    _, groups, _ = match_graphs(graphs)

    cluster_a_ids = {nt_a1.id, dv_a1.id, jc_a1.id, nt_a2.id, dv_a2.id, jc_a2.id}
    cluster_b_ids = {ql_b1.id, cs_b1.id, sp_b1.id, ql_b2.id, cs_b2.id, sp_b2.id}

    for group in groups:
        has_a = bool(group & cluster_a_ids)
        has_b = bool(group & cluster_b_ids)
        assert not (has_a and has_b), (
            f"Progressive merging caused cross-cluster false merge: {group}"
        )

    # Within-cluster merges should still work
    nt_group = _find_group_containing(groups, nt_a1.id)
    assert nt_group is not None and nt_a2.id in nt_group
    ql_group = _find_group_containing(groups, ql_b1.id)
    assert ql_group is not None and ql_b2.id in ql_group


# ---------------------------------------------------------------------------
# 5. False-merge regressions from real data
# ---------------------------------------------------------------------------


def test_shared_employee_bridge_no_company_merge():
    """A person who held the same role at two different companies should
    not cause those companies to merge, and same-name company entities
    should still merge.

    Teresa Nakamura was CFO at both Cascade Robotics and CloudScale,
    reported by two sources with different phrasings. The Nakamura pair
    bridges the companies structurally, but the CFO-event pairs have
    name-mismatched patients and suppress.

    Reproduces the Cascade Robotics / CloudScale pattern from real data."""
    g1 = Graph(id="g1")
    nak1 = g1.add_entity("Teresa Nakamura")
    cascade1 = g1.add_entity("Cascade Robotics")
    cloud1 = g1.add_entity("CloudScale")
    fact(g1, "be CFO of", agent=nak1, patients=(cascade1,))
    fact(g1, "be CFO at", agent=nak1, patients=(cloud1,))

    g2 = Graph(id="g2")
    nak2 = g2.add_entity("Teresa Nakamura")
    cascade2 = g2.add_entity("Cascade Robotics")
    cloud2 = g2.add_entity("CloudScale")
    fact(g2, "be appointed CFO of", agent=nak2, patients=(cascade2,))
    fact(g2, "be chief financial officer of", agent=nak2, patients=(cloud2,))

    graphs = [g1, g2]
    _, groups, _ = match_graphs(graphs)

    cascade_ids = {cascade1.id, cascade2.id}
    cloud_ids = {cloud1.id, cloud2.id}
    for group in groups:
        assert not (group & cascade_ids and group & cloud_ids), (
            f"Companies falsely merged via shared employee bridge: {group}"
        )

    cas_group = _find_group_containing(groups, cascade1.id)
    assert cas_group is not None and cascade2.id in cas_group
    cloud_group = _find_group_containing(groups, cloud1.id)
    assert cloud_group is not None and cloud2.id in cloud_group


def test_regulator_and_regulated_entity_stay_separate():
    """A regulatory body and the entity it regulates should not merge,
    and same-name entities should still merge across sources.

    Two outlets both report on the Data Protection Commission and
    Vantara AI. The outlets are not part of the world graph (extraction
    drops them), so the scenario reduces to both entities being patients
    of report events with mismatched agents — the cross pairs suppress.

    Reproduces the DPC / Vantara AI pattern from real data."""
    g1 = Graph(id="g1")
    dpc1 = g1.add_entity("Data Protection Commission")
    vantara1 = g1.add_entity("Vantara AI")
    fact(g1, "be fined by", agent=vantara1, patients=(dpc1,))
    fact(g1, "publish report on", agent=dpc1, patients=(vantara1,))

    g2 = Graph(id="g2")
    dpc2 = g2.add_entity("Data Protection Commission")
    vantara2 = g2.add_entity("Vantara AI")
    fact(g2, "be fined by", agent=vantara2, patients=(dpc2,))
    fact(g2, "publish report on", agent=dpc2, patients=(vantara2,))

    graphs = [g1, g2]
    _, groups, _ = match_graphs(graphs)

    dpc_ids = {dpc1.id, dpc2.id}
    vantara_ids = {vantara1.id, vantara2.id}
    for group in groups:
        assert not (group & dpc_ids and group & vantara_ids), (
            f"Regulator and regulated entity incorrectly merged: {group}"
        )

    dpc_group = _find_group_containing(groups, dpc1.id)
    assert dpc_group is not None and dpc2.id in dpc_group
    vantara_group = _find_group_containing(groups, vantara1.id)
    assert vantara_group is not None and vantara2.id in vantara_group


def test_shared_acquirer_does_not_merge_different_targets():
    """Two different acquisition targets should not merge just because
    they were acquired by the same company, even when multiple sources
    report the acquisitions with different event labels.

    A neutral article (hub only, no targets) lets the hub entity merge
    across all articles via progressive merging. After the hub merges,
    each target's only structural connection is the shared acquirer
    event — a single path, which never crosses the merge bar.

    Reproduces the Lightwave Analytics / CloudScale false merge pattern
    from real data (both acquired by Meridian Technologies)."""
    # Neutral article: hub entity with an unrelated event.
    g0 = Graph(id="g0")
    hub0 = g0.add_entity("Meridian Technologies")
    loc0 = g0.add_entity("Pittsburgh")
    fact(g0, "be based in", agent=hub0, patients=(loc0,))

    # Target-B articles (2 label variants)
    g1 = Graph(id="g1")
    hub1 = g1.add_entity("Meridian Technologies")
    b1 = g1.add_entity("Lightwave Analytics")
    fact(g1, "acquire", agent=hub1, patients=(b1,))

    g2 = Graph(id="g2")
    hub2 = g2.add_entity("Meridian Technologies")
    b2 = g2.add_entity("Lightwave Analytics")
    fact(g2, "purchase", agent=hub2, patients=(b2,))

    # Target-C articles (2 label variants)
    g3 = Graph(id="g3")
    hub3 = g3.add_entity("Meridian Technologies")
    c1 = g3.add_entity("CloudScale")
    fact(g3, "acquire", agent=hub3, patients=(c1,))

    g4 = Graph(id="g4")
    hub4 = g4.add_entity("Meridian Technologies")
    c2 = g4.add_entity("CloudScale")
    fact(g4, "purchase", agent=hub4, patients=(c2,))

    graphs = [g0, g1, g2, g3, g4]
    _, groups, _ = match_graphs(graphs)

    b_ids = {b1.id, b2.id}
    c_ids = {c1.id, c2.id}
    for group in groups:
        assert not (group & b_ids and group & c_ids), (
            f"Acquisition targets falsely merged via shared acquirer: {group}"
        )

    # Same-name merges should work (the neutral article enables this).
    b_group = _find_group_containing(groups, b1.id)
    assert b_group is not None and b2.id in b_group
    c_group = _find_group_containing(groups, c1.id)
    assert c_group is not None and c2.id in c_group
