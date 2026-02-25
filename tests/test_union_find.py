"""Unit tests for UnionFind."""

from worldgraph.match import UnionFind


def test_new_element_is_own_root():
    uf = UnionFind()
    assert uf.find("a") == "a"


def test_union_makes_same_root():
    uf = UnionFind()
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")


def test_transitivity():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")


def test_path_compression():
    uf = UnionFind()
    # Build a chain: a->b->c->d
    uf.union("a", "b")
    uf.union("b", "c")
    uf.union("c", "d")
    root = uf.find("a")
    # After find, every node on the path should point directly to root
    assert uf.parent["a"] == root
    assert uf.parent["b"] == root
    assert uf.parent["c"] == root


def test_union_by_rank_higher_rank_wins():
    uf = UnionFind()
    # Union a+b raises a's rank to 1; then union a+c should attach c under a
    uf.union("a", "b")
    uf.union("a", "c")
    assert uf.find("c") == uf.find("a")
    assert uf.parent["c"] == uf.find("a")


def test_disjoint_elements_have_different_roots():
    uf = UnionFind()
    uf.union("a", "b")
    uf.find("c")
    assert uf.find("a") != uf.find("c")


def test_union_is_idempotent():
    uf = UnionFind()
    uf.union("a", "b")
    root_before = uf.find("a")
    uf.union("a", "b")
    assert uf.find("a") == root_before


def test_works_with_tuple_keys():
    """Entity keys in the codebase are (graph_id, entity_id) tuples."""
    uf = UnionFind()
    uf.union(("g1", "e1"), ("g2", "e1"))
    uf.union(("g2", "e1"), ("g3", "e1"))
    assert uf.find(("g1", "e1")) == uf.find(("g3", "e1"))
