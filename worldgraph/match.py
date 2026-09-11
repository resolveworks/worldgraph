"""Stage 2: term alignment via damped pairwise propagation.

A graph is a set of terms — entities and statements — and statements are
about terms, so qualifiers, attribution, and claims-about-claims nest to
any depth. Matching aligns cross-graph pairs of the same kind
(entity↔entity, statement↔statement) with a single confidence score per
pair, refined by damped fixed-point iteration over per-neighbor
best-counterpart evidence (Similarity Flooding / PARIS family).

Slot alignment is the core invariant: the position a term occupies in a
statement is part of its evidence signature. A statement's subject slot
aligns only with the counterpart statement's subject slot, object with
object; a participant that is the subject of one statement finds
counterparts only among subjects of the counterpart statement. Because
extraction normalizes voice, a name-aligned participant sitting in
opposite slots of two otherwise-similar statements shows up as failed
slot-aligned counterpart searches — negative evidence for a genuine
direction disagreement.

Nesting needs no special machinery: statements about statements are
ordinary terms, so denials of the same claim lift each other through
their object slots and qualifier statements corroborate the statements
they qualify through their subject slots, by the same propagation.

Seeding: entity pairs from name similarity (names.py), statement pairs
from an injectable predicate-similarity prior (the constant neutral
prior when none is injected). Similarity only proposes — names or priors
alone never merge anything. Structure disposes: merging requires at
least one structurally tested neighbor, and the union-find maintained
during propagation is the sole merge authority.
"""

import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import click

from worldgraph.constants import MERGE_THRESHOLD, NEUTRAL_PRIOR
from worldgraph.embed import Embedder
from worldgraph.graph import (
    Entity,
    Graph,
    Statement,
    Term,
    load_graph,
    save_graph,
)
from worldgraph.names import build_idf, soft_tfidf
from worldgraph.priors import make_predicate_prior

# Term ids live in one namespace per article graph, so the matcher keys
# terms by (graph_id, term_id).
Qid = tuple[str, str]
Confidence = dict[tuple[Qid, Qid], float]
MatchGroup = list[Qid]

# The two slots a term can occupy in a statement.
SUBJECT = "subject"
OBJECT = "object"

PredicatePrior = Callable[[Statement, Statement], float]


def qualified(qid: Qid) -> str:
    """Flat string form of a qualified id, used for merged-graph term ids
    and match groups so terms from different article graphs cannot
    collide."""
    return f"{qid[0]}:{qid[1]}"


class UnionFind:
    def __init__(self):
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class Functionality(NamedTuple):
    forward: float
    inverse: float


class Neighbor(NamedTuple):
    """An entry in a term's weighted participation adjacency list.

    ``slot`` is the position the neighbor occupies in the shared
    statement (or, seen from the statement, the position the participant
    takes). ``pos_weight`` weights positive evidence, ``neg_weight``
    negative evidence, by the slot's functionality (see
    ``_build_adjacency``).
    """

    node: Qid
    slot: str
    pos_weight: float
    neg_weight: float


# ---------------------------------------------------------------------------
# Slot functionality
# ---------------------------------------------------------------------------


def compute_functionality(graphs: list[Graph]) -> dict[str, Functionality]:
    """Compute functionality and inverse functionality per slot.

    Forward functionality of a slot: given a statement, how discriminative
    is an alignment through that slot — 1 / avg number of terms occupying
    the slot per statement, which is always 1 by construction (a statement
    has exactly one subject and one object).

    Inverse functionality: given a participant in this slot, how much does
    it determine the statement — 1 / avg number of statements in which a
    participant appears in the slot. Entity participants pool by primary
    name across graphs (the same entity reported by several outlets pools
    its statistics); statement participants pool by occurrence (each
    statement is an instance).
    """
    targets_per_source: dict[str, dict[Qid, set[str]]] = {
        SUBJECT: defaultdict(set),
        OBJECT: defaultdict(set),
    }
    sources_per_target: dict[str, dict[str, set[Qid]]] = {
        SUBJECT: defaultdict(set),
        OBJECT: defaultdict(set),
    }
    for graph in graphs:
        for term in graph.terms.values():
            if not isinstance(term, Statement):
                continue
            source = (graph.id, term.id)
            for slot, endpoint in ((SUBJECT, term.subject), (OBJECT, term.object)):
                target = graph.terms[endpoint]
                key = (
                    target.names[0]
                    if isinstance(target, Entity)
                    else qualified((graph.id, target.id))
                )
                targets_per_source[slot][source].add(key)
                sources_per_target[slot][key].add(source)

    result: dict[str, Functionality] = {}
    for slot in (SUBJECT, OBJECT):
        if not targets_per_source[slot]:
            continue
        avg_out = sum(len(t) for t in targets_per_source[slot].values()) / len(
            targets_per_source[slot]
        )
        avg_in = sum(len(s) for s in sources_per_target[slot].values()) / len(
            sources_per_target[slot]
        )
        result[slot] = Functionality(1.0 / avg_out, 1.0 / avg_in)
    return result


# ---------------------------------------------------------------------------
# Unified term universe and adjacency
# ---------------------------------------------------------------------------


def qualified_terms(graphs: list[Graph]) -> dict[Qid, Term]:
    """Combine article graphs into one qualified term universe. Ids are
    per-graph namespaces, so qualified ids are unique — duplicate graph
    ids are invalid input and raise."""
    seen: set[str] = set()
    for graph in graphs:
        if graph.id in seen:
            raise ValueError(f"duplicate graph id: {graph.id!r}")
        seen.add(graph.id)
    return {
        (graph.id, term_id): term
        for graph in graphs
        for term_id, term in graph.terms.items()
    }


def _dedup_neighbors(neighbors: list[Neighbor]) -> list[Neighbor]:
    """Deduplicate adjacency entries by (neighbor, slot) — the same
    structural evidence — keeping the max-weight entry per group, in a
    deterministic order."""
    best: dict[tuple[Qid, str], Neighbor] = {}
    for nbr in sorted(neighbors, key=lambda n: (n.node, n.slot, -n.pos_weight)):
        best.setdefault((nbr.node, nbr.slot), nbr)
    return [best[key] for key in sorted(best)]


def _build_adjacency(
    terms: dict[Qid, Term],
    functionality: dict[str, Functionality],
) -> dict[Qid, list[Neighbor]]:
    """Build the participation adjacency from statement endpoints.

    Each statement contributes two adjacency entries per endpoint — one
    on the statement, one on the participant — carrying the slot the
    participant occupies. Evidence from a statement toward a participant
    is weighted by the slot's inverse functionality (how much this
    participant determines the statement); evidence from a participant
    toward its statements by forward functionality (a statement
    determines its slot occupants exactly). Negative weights mirror this.
    """
    adjacency: dict[Qid, list[Neighbor]] = defaultdict(list)
    for qid, term in terms.items():
        if not isinstance(term, Statement):
            continue
        for slot, endpoint in ((SUBJECT, term.subject), (OBJECT, term.object)):
            participant = (qid[0], endpoint)
            fwd, inv = functionality[slot]
            adjacency[qid].append(
                Neighbor(participant, slot, pos_weight=inv, neg_weight=fwd)
            )
            adjacency[participant].append(
                Neighbor(qid, slot, pos_weight=fwd, neg_weight=inv)
            )
    return {node: _dedup_neighbors(nbrs) for node, nbrs in adjacency.items()}


def _build_pairs(terms: dict[Qid, Term]) -> list[tuple[Qid, Qid]]:
    """Cross-graph, same-kind term pairs. Entities and statements live in
    different worlds — a pair only ever spans graphs, never kinds."""
    pairs: list[tuple[Qid, Qid]] = []
    qids = sorted(terms)
    for i, a in enumerate(qids):
        for b in qids[i + 1 :]:
            if a[0] == b[0]:
                continue
            if isinstance(terms[a], Statement) != isinstance(terms[b], Statement):
                continue
            pairs.append((a, b))
    return pairs


def _seed_confidence(
    terms: dict[Qid, Term],
    idf: dict[str, float],
    pairs: list[tuple[Qid, Qid]],
    predicate_prior: PredicatePrior | None,
) -> tuple[Confidence, Confidence]:
    """Seed confidence: entity pairs from name similarity (Soft TF-IDF +
    Jaro-Winkler over name lists), statement pairs from the
    predicate-similarity prior (the constant neutral prior when none is
    injected). The prior is consulted once per pair, in sorted id order.

    Returns (conf, baseline) where ``baseline`` is the read-only anchor
    for the seed-as-baseline update formula.
    """
    conf: Confidence = {}
    baseline: Confidence = {}
    for a, b in pairs:
        term_a, term_b = terms[a], terms[b]
        if isinstance(term_a, Statement):
            assert isinstance(term_b, Statement)
            if predicate_prior is None:
                seed = NEUTRAL_PRIOR
            else:
                seed = predicate_prior(term_a, term_b)
                if not 0.0 <= seed <= 1.0:
                    raise ValueError(
                        "predicate prior out of [0, 1] for "
                        f"({term_a.predicate!r}, {term_b.predicate!r}): {seed}"
                    )
        else:
            assert isinstance(term_a, Entity) and isinstance(term_b, Entity)
            seed = max(
                soft_tfidf(na, nb, idf) for na in term_a.names for nb in term_b.names
            )
        conf[(a, b)] = seed
        conf[(b, a)] = seed
        baseline[(a, b)] = seed
        baseline[(b, a)] = seed
    return conf, baseline


def _remap_confidence(conf: Confidence, uf: UnionFind) -> Confidence:
    """Remap a confidence dict to canonical reps, taking max on collisions."""
    remapped: Confidence = {}
    for (a, b), score in conf.items():
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        old = remapped.get((ra, rb), 0.0)
        remapped[(ra, rb)] = max(old, score)
        remapped[(rb, ra)] = max(old, score)
    return remapped


# ---------------------------------------------------------------------------
# Similarity propagation
# ---------------------------------------------------------------------------


def propagate_similarity(
    terms: dict[Qid, Term],
    idf: dict[str, float],
    functionality: dict[str, Functionality],
    *,
    predicate_prior: PredicatePrior | None = None,
    max_iter: int = 30,
    epsilon: float = 1e-4,
    exp_lambda: float = 2.0,
    merge_threshold: float = MERGE_THRESHOLD,
    damping: float = 0.5,
    prior_strength: float = 1.0,
) -> tuple[Confidence, dict[Qid, list[Qid]]]:
    """Run damped similarity propagation with progressive merging.

    A single confidence score per term pair integrates both positive and
    negative structural evidence. For each slot-aligned neighbor of term
    A, we find its best counterpart among B's neighbors occupying the
    same slot and contribute once — positive if the best counterpart
    confidence exceeds the neutral prior, negative if it falls below. A
    counterpart sitting exactly at the neutral prior is untested: it
    contributes nothing and does not count toward the tested-neighbor
    total. Neighbors that resolve to either term in the pair are excluded
    to prevent circular self-reference.

    Evidence is computed **bidirectionally** (A→B and B→A) and averaged.
    Positive evidence is weighted by **Bayesian shrinkage**
    ``n / (n + κ)`` where ``n`` is the number of neighbors that found a
    slot-aligned counterpart and ``κ`` (``prior_strength``) controls how
    many tested neighbors are needed before trusting structural matches —
    this shrinkage is what keeps single-path pairs below the merge bar.
    Negative evidence has full weight — a mismatch is decisive and does
    not need corroboration.

    Each directional score is::

        weight = n_tested / (n_tested + prior_strength)
        evidence = pos_agg * (1 - baseline) * weight - neg_agg * baseline
        computed_dir = baseline + evidence

    The final value is the average of both directions, blended with the
    previous score via damping. The ``baseline`` is the name-similarity
    seed for entity pairs and the predicate prior for statement pairs:
    positive evidence can only spend the headroom above it; negative
    evidence pushes below it.

    On merge, the canonical adjacency for the new representative is built
    by combining and deduplicating the adjacency lists of the merged
    terms — O(degree) per merge, not O(|statements|).

    Returns (confidence, members). The union-find maintained during
    propagation is the **sole merge authority**: a pair merges iff it
    crossed ``merge_threshold`` with at least one tested neighbor, and
    ``members`` maps every canonical representative to its member terms
    (singletons included). There is no second, post-hoc grouping pass.
    """
    uf = UnionFind()
    for qid in terms:
        uf.find(qid)

    canonical_adj = _build_adjacency(terms, functionality)
    pairs = _build_pairs(terms)

    if not pairs:
        return {}, {qid: [qid] for qid in terms}

    conf, baseline = _seed_confidence(terms, idf, pairs, predicate_prior)

    def _directional_evidence(
        src: Qid,
        tgt: Qid,
        prev: Confidence,
    ) -> tuple[float, float, int]:
        """Per-neighbor best-counterpart evidence from src's perspective.

        For each slot-aligned neighbor of *src*, find its best counterpart
        among *tgt's* neighbors in the same slot and contribute positive
        (nc > neutral) or negative (nc < neutral) evidence once. A
        counterpart at exactly the neutral prior is untested: no
        contribution.

        Returns (pos_strength, neg_strength, n_with_counterpart).
        """
        pos_strength = 0.0
        neg_strength = 0.0
        n_with_counterpart = 0
        nbrs_tgt = canonical_adj.get(tgt, [])
        for nbr_s in canonical_adj.get(src, []):
            if nbr_s.node == src or nbr_s.node == tgt:
                continue

            best_nc: float | None = None
            best_pos_w = 0.0
            best_neg_w = 0.0
            for nbr_t in nbrs_tgt:
                if nbr_t.slot != nbr_s.slot:
                    continue
                if nbr_t.node == tgt or nbr_t.node == src:
                    continue
                nc = (
                    1.0
                    if nbr_s.node == nbr_t.node
                    else prev.get((nbr_s.node, nbr_t.node), 0.0)
                )
                if best_nc is None or nc > best_nc:
                    best_nc = nc
                    best_pos_w = min(nbr_s.pos_weight, nbr_t.pos_weight)
                    best_neg_w = min(nbr_s.neg_weight, nbr_t.neg_weight)

            if best_nc is None:
                continue

            if best_nc > NEUTRAL_PRIOR:
                pos_strength += best_pos_w * best_nc
                n_with_counterpart += 1
            else:
                neg_nc = 1.0 - best_nc
                if neg_nc > NEUTRAL_PRIOR:
                    neg_strength += best_neg_w * neg_nc
                    n_with_counterpart += 1

        return pos_strength, neg_strength, n_with_counterpart

    n_tested: dict[tuple[Qid, Qid], int] = {}

    for _ in range(max_iter):
        prev = dict(conf)
        changed = False

        for ca, cb in pairs:
            pos_fwd, neg_fwd, n_cp_fwd = _directional_evidence(ca, cb, prev)
            pos_bwd, neg_bwd, n_cp_bwd = _directional_evidence(cb, ca, prev)

            n_tested[(ca, cb)] = n_cp_fwd + n_cp_bwd
            seed = baseline[(ca, cb)]

            # Bidirectional: shrinkage-weighted average of both perspectives.
            # Shrinkage applies only to positive evidence (structural
            # matches need corroboration); negative evidence has full
            # weight (a mismatch is decisive).
            dir_sum = 0.0
            for pos_s, neg_s, n_cp in (
                (pos_fwd, neg_fwd, n_cp_fwd),
                (pos_bwd, neg_bwd, n_cp_bwd),
            ):
                w = n_cp / (n_cp + prior_strength) if n_cp > 0 else 0.0
                pa = 1.0 - math.exp(-exp_lambda * pos_s) if pos_s > 0 else 0.0
                na = 1.0 - math.exp(-exp_lambda * neg_s) if neg_s > 0 else 0.0
                evidence = pa * (1.0 - seed) * w - na * seed
                dir_sum += max(0.0, min(1.0, seed + evidence))
            computed = dir_sum / 2

            old = prev[(ca, cb)]
            new_val = (1.0 - damping) * old + damping * computed
            conf[(ca, cb)] = new_val
            conf[(cb, ca)] = new_val

            if abs(new_val - old) > epsilon:
                changed = True

        if changed:
            continue

        # --- Progressive merging (requires structural evidence) ---
        new_merges = [
            (ca, cb)
            for ca, cb in pairs
            if conf[(ca, cb)] >= merge_threshold
            and uf.find(ca) != uf.find(cb)
            and n_tested.get((ca, cb), 0) > 0
        ]

        if new_merges:
            for ca, cb in new_merges:
                uf.union(ca, cb)

            # Update canonical_adj incrementally: combine + dedup.
            merge_groups: dict[Qid, list[Qid]] = defaultdict(list)
            for e in sorted({e for pair in new_merges for e in pair}):
                merge_groups[uf.find(e)].append(e)
            for new_canon, old_canons in merge_groups.items():
                combined: list[Neighbor] = []
                for oc in old_canons:
                    combined.extend(canonical_adj.get(oc, []))
                remapped = [
                    Neighbor(
                        uf.find(nbr.node),
                        nbr.slot,
                        nbr.pos_weight,
                        nbr.neg_weight,
                    )
                    for nbr in combined
                    if uf.find(nbr.node) != new_canon
                ]
                canonical_adj[new_canon] = _dedup_neighbors(remapped)

            # Remap pairs and confidence dicts to canonical reps.
            pair_set: set[tuple[Qid, Qid]] = set()
            new_pairs: list[tuple[Qid, Qid]] = []
            for a, b in pairs:
                ra, rb = uf.find(a), uf.find(b)
                if ra == rb:
                    continue
                pair = (min(ra, rb), max(ra, rb))
                if pair not in pair_set:
                    pair_set.add(pair)
                    new_pairs.append(pair)
            pairs = new_pairs

            conf = _remap_confidence(conf, uf)
            baseline = _remap_confidence(baseline, uf)

            if not pairs:
                break
            continue

        # Converged, no new merges — done.
        break

    # Every term grouped under its canonical representative.
    members: dict[Qid, list[Qid]] = defaultdict(list)
    for qid in terms:
        members[uf.find(qid)].append(qid)

    # Expand canonical-rep confidence to original term pairs.
    final: Confidence = {}
    for (ca, cb), score in conf.items():
        if ca == cb:
            continue
        for ma in members.get(ca, [ca]):
            for mb in members.get(cb, [cb]):
                if ma[0] == mb[0]:
                    continue
                old = final.get((ma, mb), 0.0)
                new_val = max(old, score)
                final[(ma, mb)] = new_val
                final[(mb, ma)] = new_val

    # Merged pairs get 1.0 — they were committed with high confidence.
    for group_members in members.values():
        if len(group_members) < 2:
            continue
        for i, ma in enumerate(group_members):
            for mb in group_members[i + 1 :]:
                if ma[0] == mb[0]:
                    continue
                final[(ma, mb)] = 1.0
                final[(mb, ma)] = 1.0

    return final, dict(members)


# ---------------------------------------------------------------------------
# Merged output graph
# ---------------------------------------------------------------------------


def build_merged_graph(terms: dict[Qid, Term], members: dict[Qid, list[Qid]]) -> Graph:
    """Build the merged canonical graph: one term per union-find group.

    Merged entities carry the union of their members' names as aliases
    (the representative's names first, then members' aliases in sorted
    member order); merged statements appear once, with the
    representative's predicate and endpoints remapped to the canonical
    ids of their participants' groups. Term ids are the qualified
    ``graph_id:term_id`` of the representative, and ``graph_id`` records
    the representative's origin graph.
    """
    canonical_of: dict[Qid, Qid] = {
        member: canonical for canonical, ms in members.items() for member in ms
    }
    merged = Graph(id="merged")
    for canonical in sorted(members):
        rep = terms[canonical]
        if isinstance(rep, Entity):
            names = list(rep.names)
            for member in sorted(members[canonical]):
                member_term = terms[member]
                assert isinstance(member_term, Entity)
                for name in member_term.names:
                    if name not in names:
                        names.append(name)
            term: Term = Entity(
                id=qualified(canonical), graph_id=canonical[0], names=names
            )
        else:
            term = Statement(
                id=qualified(canonical),
                graph_id=canonical[0],
                subject=qualified(canonical_of[(canonical[0], rep.subject)]),
                predicate=rep.predicate,
                object=qualified(canonical_of[(canonical[0], rep.object)]),
            )
        merged.terms[term.id] = term
    merged.validate()
    return merged


# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------


def match_graphs(
    graphs: list[Graph],
    *,
    predicate_prior: PredicatePrior | None = None,
    **propagate_kwargs,
) -> tuple[Confidence, list[MatchGroup], Graph]:
    """Core matching pipeline: graphs → (confidence, match groups, merged graph).

    Computes name IDF and slot functionality, then runs similarity
    propagation. ``predicate_prior`` seeds statement pairs with predicate
    similarity (None means the constant neutral prior). Match groups come
    straight from the union-find inside propagation — the sole merge
    authority (one threshold, gated on structural evidence).
    """
    terms = qualified_terms(graphs)

    entity_names = [
        name
        for term in terms.values()
        if isinstance(term, Entity)
        for name in term.names
    ]
    idf = build_idf(entity_names)
    functionality = compute_functionality(graphs)

    confidence, members = propagate_similarity(
        terms,
        idf,
        functionality,
        predicate_prior=predicate_prior,
        **propagate_kwargs,
    )
    groups = sorted(sorted(ms) for ms in members.values() if len(ms) > 1)
    merged = build_merged_graph(terms, members)
    return confidence, groups, merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _label(term: Term) -> str:
    """Human label for group echo: primary name for entities, predicate
    for statements."""
    return term.names[0] if isinstance(term, Entity) else term.predicate


def run_matching(
    graph_files: list[Path],
    output_path: Path,
    merge_threshold: float = MERGE_THRESHOLD,
    max_iter: int = 30,
    epsilon: float = 1e-4,
    embedder: Embedder | None = None,
) -> None:
    """Load graphs, run matching pipeline, save the merged graph with
    match groups.

    Statement pairs seed from a predicate prior built from *embedder*;
    ``None`` builds the real one from ``EMBEDDING_MODEL`` (raises if
    unset). The seam for tests and experiments is an injected embedder —
    the pipeline always runs with priors.
    """
    graphs = [load_graph(path) for path in graph_files]
    terms = qualified_terms(graphs)
    click.echo(f"Loaded {len(graphs)} graphs")
    for graph in graphs:
        n_entities = sum(1 for t in graph.terms.values() if isinstance(t, Entity))
        n_statements = sum(
            1 for t in graph.terms.values() if isinstance(t, Statement)
        )
        click.echo(f"  {graph.id}: {n_entities} entities, {n_statements} statements")

    predicate_prior = make_predicate_prior(graphs, embedder=embedder)
    _confidence, groups, merged = match_graphs(
        graphs,
        predicate_prior=predicate_prior,
        max_iter=max_iter,
        epsilon=epsilon,
        merge_threshold=merge_threshold,
    )
    matches = [[qualified(member) for member in group] for group in groups]
    save_graph(merged, output_path, matches)

    click.echo(f"\n{len(groups)} match groups:")
    for group in groups:
        labels = sorted({_label(terms[m]) for m in group})
        click.echo(f"  {' / '.join(labels)}")

    click.echo(f"\nWrote {output_path}")
