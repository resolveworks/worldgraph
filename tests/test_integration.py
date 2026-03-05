"""Layer 3 integration tests.

These tests exercise the full matching pipeline: multiple graphs → unified
graph → propagate → union-find → match groups.  They verify end-to-end
correctness on multi-source scenarios that L2 tests don't cover:

- Transitive merging across 3+ sources via union-find
- Cross-cluster isolation (independent events don't merge, even with
  identical entity names or isomorphic structure)
- Cross-event entity linking (shared entity across clusters)
"""

from collections import defaultdict

from worldgraph.constants import NAME_EDGE, RELATION_TEMPLATE
from worldgraph.embed import Embedder
from worldgraph.graph import Graph, LiteralNode
from worldgraph.match import (
    UnionFind,
    build_unified_graph,
    compute_functionality,
    propagate,
)
from worldgraph.names import build_idf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_full_pipeline(
    graphs: list[Graph],
    embedder: Embedder,
    labels: list[str],
    relations: list[str],
    match_threshold: float = 0.8,
    **propagate_kwargs,
) -> list[set[str]]:
    """Run the full pipeline and return match groups as sets of entity IDs."""
    unified = build_unified_graph(graphs)
    idf = build_idf(labels)
    rel_embs = embedder.embed([*relations, NAME_EDGE], template=RELATION_TEMPLATE)
    func = compute_functionality(graphs, rel_embs)
    confidence = propagate(unified, idf, rel_embs, func, **propagate_kwargs)

    uf = UnionFind()
    for (id_a, id_b), score in confidence.items():
        if score >= match_threshold:
            uf.union(id_a, id_b)

    entity_ids = [
        nid for nid, n in unified.nodes.items() if not isinstance(n, LiteralNode)
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for eid in entity_ids:
        groups[uf.find(eid)].append(eid)
    return [set(members) for members in groups.values() if len(members) > 1]


def _find_group_containing(groups: list[set[str]], entity_id: str) -> set[str] | None:
    """Find the match group containing a given entity ID."""
    for g in groups:
        if entity_id in g:
            return g
    return None


# ---------------------------------------------------------------------------
# 1. Multi-source transitive merging
# ---------------------------------------------------------------------------


def test_three_source_with_person_name_variation(embedder):
    """Three sources with person name variations, each with enough
    structural context for propagation to work.

    "Dr. Priya Sharma" / "Priya Sharma" / "Dr. Sharma" — the name
    similarity alone (~0.75) is below match threshold. Multiple shared
    neighbors with identical names (Meridian, James Chen, Stanford)
    provide structural evidence to bridge the gap."""
    g1 = Graph(id="article-1")
    m1 = g1.add_entity("Meridian Technologies")
    p1 = g1.add_entity("Dr. Priya Sharma")
    j1 = g1.add_entity("James Chen")
    su1 = g1.add_entity("Stanford University")
    dv1 = g1.add_entity("DataVault Inc")
    g1.add_edge(m1, p1, "hired")
    g1.add_edge(p1, j1, "collaborates with")
    g1.add_edge(p1, su1, "alumna of")
    g1.add_edge(m1, dv1, "acquired")

    g2 = Graph(id="article-2")
    m2 = g2.add_entity("Meridian Technologies")
    p2 = g2.add_entity("Priya Sharma")
    j2 = g2.add_entity("James Chen")
    su2 = g2.add_entity("Stanford University")
    dv2 = g2.add_entity("DataVault Inc")
    g2.add_edge(m2, p2, "hired")
    g2.add_edge(p2, j2, "collaborates with")
    g2.add_edge(p2, su2, "alumna of")
    g2.add_edge(m2, dv2, "acquired")

    g3 = Graph(id="article-3")
    m3 = g3.add_entity("Meridian Technologies")
    p3 = g3.add_entity("Dr. Sharma")
    j3 = g3.add_entity("James Chen")
    su3 = g3.add_entity("Stanford University")
    dv3 = g3.add_entity("DataVault Inc")
    g3.add_edge(m3, p3, "hired")
    g3.add_edge(p3, j3, "collaborates with")
    g3.add_edge(p3, su3, "alumna of")
    g3.add_edge(m3, dv3, "acquired")

    labels = [
        "Meridian Technologies",
        "Dr. Priya Sharma",
        "Priya Sharma",
        "Dr. Sharma",
        "James Chen",
        "Stanford University",
        "DataVault Inc",
    ]
    relations = ["hired", "collaborates with", "alumna of", "acquired"]

    groups = _run_full_pipeline([g1, g2, g3], embedder, labels, relations)

    m_group = _find_group_containing(groups, m1.id)
    assert m_group is not None, "Meridian entities not merged"
    assert m2.id in m_group and m3.id in m_group

    # p1-p2 and p1-p3 should each merge (name sim ~0.75 + structural
    # evidence from 3 shared neighbors). p2-p3 may not merge directly
    # (name sim 0.28) but union-find transitivity through p1 links all three.
    p_group = _find_group_containing(groups, p1.id)
    assert p_group is not None, "Sharma entities not merged"
    assert p2.id in p_group and p3.id in p_group


# ---------------------------------------------------------------------------
# 2. Cross-cluster isolation
# ---------------------------------------------------------------------------


def test_identical_names_different_contexts_no_merge(embedder):
    """Two different people with identical names in unrelated clusters.

    Cluster A: Dr. James Chen leads Advanced AI Lab, funded by NSF
    Cluster B: Dr. James Chen leads Climate Research Lab, funded by EPA

    Name similarity is 1.0 and both have the same relation ("leads"),
    same structure shape, and similar funder relation. But the actual
    neighbors are completely different — no structural evidence supports
    merging the two James Chens."""
    # Cluster A: AI research
    a1 = Graph(id="ai-1")
    jc_a1 = a1.add_entity("Dr. James Chen")
    lab_a1 = a1.add_entity("Advanced AI Lab")
    nsf_a1 = a1.add_entity("National Science Foundation")
    a1.add_edge(jc_a1, lab_a1, "leads")
    a1.add_edge(lab_a1, nsf_a1, "funded by")

    a2 = Graph(id="ai-2")
    jc_a2 = a2.add_entity("Dr. James Chen")
    lab_a2 = a2.add_entity("Advanced AI Lab")
    nsf_a2 = a2.add_entity("National Science Foundation")
    a2.add_edge(jc_a2, lab_a2, "leads")
    a2.add_edge(lab_a2, nsf_a2, "funded by")

    # Cluster B: climate research — same name, same structure, different entities
    b1 = Graph(id="climate-1")
    jc_b1 = b1.add_entity("Dr. James Chen")
    lab_b1 = b1.add_entity("Climate Research Lab")
    epa_b1 = b1.add_entity("Environmental Protection Agency")
    b1.add_edge(jc_b1, lab_b1, "leads")
    b1.add_edge(lab_b1, epa_b1, "funded by")

    b2 = Graph(id="climate-2")
    jc_b2 = b2.add_entity("Dr. James Chen")
    lab_b2 = b2.add_entity("Climate Research Lab")
    epa_b2 = b2.add_entity("Environmental Protection Agency")
    b2.add_edge(jc_b2, lab_b2, "leads")
    b2.add_edge(lab_b2, epa_b2, "funded by")

    labels = [
        "Dr. James Chen",
        "Advanced AI Lab",
        "National Science Foundation",
        "Climate Research Lab",
        "Environmental Protection Agency",
    ]
    relations = ["leads", "funded by"]

    groups = _run_full_pipeline([a1, a2, b1, b2], embedder, labels, relations)

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
# 3. Cross-event entity linking
# ---------------------------------------------------------------------------


def test_shared_entity_across_clusters(embedder):
    """An entity appearing in two independent event clusters should be
    linked across them, while event-specific entities stay isolated.

    Cluster A: Meridian Technologies acquired DataVault (CEO: Elena Vasquez)
    Cluster B: Meridian Technologies settles FTC investigation (CEO: Elena Vasquez)

    Meridian and Elena Vasquez are the shared entities — they appear with
    identical names and shared structural context (the CEO relation) across
    both clusters. DataVault and FTC should NOT merge."""
    # Cluster A: acquisition (2 sources)
    a1 = Graph(id="acq-1")
    m_a1 = a1.add_entity("Meridian Technologies")
    dv_a1 = a1.add_entity("DataVault")
    ev_a1 = a1.add_entity("Elena Vasquez")
    a1.add_edge(m_a1, dv_a1, "acquired")
    a1.add_edge(m_a1, ev_a1, "CEO is")

    a2 = Graph(id="acq-2")
    m_a2 = a2.add_entity("Meridian Technologies")
    dv_a2 = a2.add_entity("DataVault")
    ev_a2 = a2.add_entity("Elena Vasquez")
    a2.add_edge(m_a2, dv_a2, "purchased")
    a2.add_edge(m_a2, ev_a2, "CEO is")

    # Cluster B: FTC investigation (2 sources) — shares Meridian + Elena
    b1 = Graph(id="ftc-1")
    m_b1 = b1.add_entity("Meridian Technologies")
    ftc_b1 = b1.add_entity("Federal Trade Commission")
    ev_b1 = b1.add_entity("Elena Vasquez")
    b1.add_edge(ftc_b1, m_b1, "investigates")
    b1.add_edge(m_b1, ev_b1, "CEO is")

    b2 = Graph(id="ftc-2")
    m_b2 = b2.add_entity("Meridian Technologies")
    ftc_b2 = b2.add_entity("Federal Trade Commission")
    ev_b2 = b2.add_entity("Elena Vasquez")
    b2.add_edge(ftc_b2, m_b2, "investigates")
    b2.add_edge(m_b2, ev_b2, "CEO is")

    labels = [
        "Meridian Technologies",
        "DataVault",
        "Elena Vasquez",
        "Federal Trade Commission",
    ]
    relations = ["acquired", "purchased", "CEO is", "investigates"]

    groups = _run_full_pipeline([a1, a2, b1, b2], embedder, labels, relations)

    # All four Meridian entities should be in one group
    m_group = _find_group_containing(groups, m_a1.id)
    assert m_group is not None, "Meridian entities not merged"
    assert {m_a1.id, m_a2.id, m_b1.id, m_b2.id} <= m_group, (
        f"Not all Meridian entities merged across clusters: {m_group}"
    )

    # DataVault should NOT merge with FTC
    dv_ids = {dv_a1.id, dv_a2.id}
    ftc_ids = {ftc_b1.id, ftc_b2.id}
    for group in groups:
        assert not (group & dv_ids and group & ftc_ids), (
            f"DataVault and FTC incorrectly merged: {group}"
        )


def test_shared_person_across_clusters(embedder):
    """A person entity shared across two event clusters, linked by
    identical names AND shared structural context (Stanford University).

    Cluster A: Elena Vasquez is CEO of Meridian Technologies, alumna of Stanford
    Cluster B: Elena Vasquez keynotes Global Tech Summit, alumna of Stanford

    The shared "Stanford University" neighbor provides structural evidence
    for cross-cluster Elena linking. Meridian and Summit should NOT merge."""
    a1 = Graph(id="hire-1")
    m1 = a1.add_entity("Meridian Technologies")
    ev1 = a1.add_entity("Elena Vasquez")
    dv1 = a1.add_entity("DataVault Inc")
    su1 = a1.add_entity("Stanford University")
    a1.add_edge(m1, ev1, "CEO is")
    a1.add_edge(m1, dv1, "acquired")
    a1.add_edge(ev1, su1, "alumna of")

    a2 = Graph(id="hire-2")
    m2 = a2.add_entity("Meridian Technologies")
    ev2 = a2.add_entity("Elena Vasquez")
    dv2 = a2.add_entity("DataVault Inc")
    su2 = a2.add_entity("Stanford University")
    a2.add_edge(m2, ev2, "CEO is")
    a2.add_edge(m2, dv2, "acquired")
    a2.add_edge(ev2, su2, "alumna of")

    b1 = Graph(id="summit-1")
    ev3 = b1.add_entity("Elena Vasquez")
    summit1 = b1.add_entity("Global Tech Summit")
    su3 = b1.add_entity("Stanford University")
    b1.add_edge(ev3, summit1, "keynotes")
    b1.add_edge(ev3, su3, "alumna of")

    b2 = Graph(id="summit-2")
    ev4 = b2.add_entity("Elena Vasquez")
    summit2 = b2.add_entity("Global Tech Summit")
    su4 = b2.add_entity("Stanford University")
    b2.add_edge(ev4, summit2, "keynotes")
    b2.add_edge(ev4, su4, "alumna of")

    labels = [
        "Meridian Technologies",
        "Elena Vasquez",
        "DataVault Inc",
        "Global Tech Summit",
        "Stanford University",
    ]
    relations = ["CEO is", "acquired", "keynotes", "alumna of"]

    groups = _run_full_pipeline([a1, a2, b1, b2], embedder, labels, relations)

    # All four Elena Vasquez entities should merge (within + across clusters)
    ev_group = _find_group_containing(groups, ev1.id)
    assert ev_group is not None, "Elena Vasquez entities not merged"
    assert {ev1.id, ev2.id, ev3.id, ev4.id} <= ev_group, (
        "Elena Vasquez not linked across clusters"
    )

    # Meridian and Summit should NOT merge
    m_ids = {m1.id, m2.id}
    summit_ids = {summit1.id, summit2.id}
    for group in groups:
        assert not (group & m_ids and group & summit_ids), (
            f"Meridian and Summit incorrectly merged: {group}"
        )
