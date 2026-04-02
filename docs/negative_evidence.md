# Negative Structural Evidence

Standard similarity propagation only accumulates positive evidence: shared neighbors with matching relations push entity-pair similarity upward. But the *absence* of expected matches can also be informative. If two entities share some neighbors but differ on a highly functional relation, that's evidence they are *not* the same entity.

This document explains the concept, PARIS's approach (and why it failed in practice), and our proposed adaptation.

## The problem

Consider two entities that name-match well: "Meridian Technologies" in article A and "Meridian Technologies" in article B. Positive propagation sees their shared neighbors and raises their similarity. But suppose:

- In article A: Meridian Technologies → acquired → DataVault Inc
- In article B: Meridian Technologies → acquired → SkyBridge Analytics

If "acquired" is a highly functional relation (each company typically acquires different targets), the fact that the targets *don't match* is evidence against the entities being the same. Without negative evidence, we rely entirely on positive signals and a threshold — which may not be enough to reject false matches between common entity names.

This matters most for:
- **Common names** that appear in unrelated event clusters (e.g. "National Bank", "John Smith")
- **Structural templates** where unrelated events produce similar graph shapes (two different acquisitions both have acquirer → acquired → target patterns)

## PARIS's approach (Equation 7)

PARIS (Suchanek et al. 2011) defines negative evidence in Equation 6. For each relation `r(x, y)`, it checks whether `y` matches any neighbor `y'` of `x'`:

```
Pr2(x ≡ x') = PRODUCT_{r(x,y)} (1 - fun(r) × PRODUCT_{r(x',y')} (1 - Pr(y ≡ y')))
```

The combined score (Equation 7) multiplies positive and negative:

```
Pr3(x ≡ x') = Pr1(x ≡ x') × Pr2(x ≡ x')
```

### Key asymmetry: which functionality?

Positive evidence uses **inverse** functionality: "this target uniquely identifies its source" — if targets match and the relation is inversely functional, the sources probably match.

Negative evidence uses **forward** functionality: "this source should have a specific target" — if the source is supposed to map to one unique target via this relation, and that target doesn't match anything, the sources probably don't match.

This asymmetry is principled. Forward functionality measures "how many targets does a typical source have?" If the answer is one (high forward functionality), then a missing target match is damning. If the answer is many (low forward functionality, e.g. `located_in`), a missing match means nothing.

### Why PARIS abandoned it

Section 6.3 of the PARIS paper reports that negative evidence was too aggressive in practice. On the restaurant dataset, using Equation 7 caused PARIS to "give up all matches between restaurants" because entities had slightly different attribute values (e.g. phone formatting: "213/467-1108" vs "213-467-1108"). A single unmatched functional relation kills the score.

The fundamental problem: **negative evidence assumes data completeness and consistency.** If `r(x, y)` exists in one graph, the absence of a matching `r(x', y')` in another graph could mean:
1. `x ≠ x'` (true negative — the entities are different)
2. The second article didn't mention this fact (incomplete coverage)
3. The fact is expressed differently and the relation phrases didn't match (false negative from relation similarity)

In knowledge base alignment (PARIS's domain), completeness is somewhat reasonable — DBpedia and YAGO are curated. In news article graphs, completeness is never reasonable. Each article covers a tiny slice of the event.

## FLORA's position

FLORA (Peng et al. 2025) explicitly excludes negation from its framework. The "Simple Positive FIS" (Definition 1) requires all variables to be non-decreasing, which is what makes the Knaster-Tarski convergence proof work. Allowing scores to decrease would break monotonicity and void the convergence guarantee.

By switching from Knaster-Tarski (monotone updates) to Banach (contraction mappings) as our convergence framework, this restriction is lifted — scores can decrease, and negative evidence integrates naturally into each iteration. See [similarity_flooding.md](similarity_flooding.md) for the full theoretical comparison.

## Our approach: integrated negative evidence via damped iteration

We need negative evidence but cannot afford PARIS's brittleness. The key insight is that negative evidence should be **weaker and more selective** than positive evidence, reflecting the fundamental asymmetry in our setting:

- A match between neighbors is *reliable* positive evidence (two articles independently reporting the same fact)
- A *missing* match could mean many things (incomplete coverage, relation phrasing mismatch, extraction error)

### How it works

Positive and negative evidence are computed together in each propagation step, feeding into a single score per entity pair. For each pair `(a, b)`, we examine each neighbor `y` of `a` and find its **best counterpart** among `b`'s neighbors connected via similar relations:

- **Positive**: if the best counterpart's confidence is above 0.5 (likely match), `y` contributes to `pos_strength`, weighted by functionality — a matching neighbor on a functional relation is strong evidence FOR the match.
- **Negative**: if the best counterpart's confidence is below 0.5 (no good match), `y` contributes to `neg_strength`, weighted by functionality — a functional relation whose target has no counterpart is evidence AGAINST the match.
- **No counterpart**: if `y` has no relation-similar neighbors on `b`'s side at all, it contributes nothing. However, the number of neighbors that *do* find counterparts determines how much positive evidence is trusted (see Bayesian shrinkage below).

Each neighbor contributes exactly once, based on its best counterpart. This avoids the all-pairs pitfall where a neighbor that matches well with one counterpart also generates bogus negative evidence from unrelated cross-pairs. For example, if `a` has neighbors Park and Chen both via "is CEO of", and `b` also has Park and Chen, the all-pairs approach would count Park₁↔Chen₂ as negative evidence despite Park₁ having a perfect match in Park₂. The per-neighbor approach correctly identifies Park₁'s best counterpart as Park₂ and contributes only positive evidence.

Neighbors that resolve to either entity in the pair are excluded to prevent **circular self-reference** — an intra-graph edge between `a` and `b` (e.g. "appeared to speak with", "is CEO of") must not serve as evidence that `a` and `b` are the same entity.

### Bidirectional evaluation with Bayesian shrinkage

Evidence is computed from **both perspectives** — `a`'s neighbors seeking counterparts among `b`'s, and `b`'s neighbors seeking counterparts among `a`'s — and the two directional scores are averaged. This ensures the result does not depend on arbitrary pair ordering.

Positive and negative evidence are treated asymmetrically. **Positive evidence** (structural matches) is weighted by a Bayesian shrinkage factor `n / (n + κ)`, where `n` is the number of neighbors that found a same-cluster counterpart and `κ` (`prior_strength`) controls how many tested neighbors are needed before trusting structural matches. **Negative evidence** (structural mismatches) has full weight — a mismatch on a functional relation is decisive and does not need corroboration.

This asymmetry reflects that name similarity alone should never be sufficient for merging. Structural evidence is needed to *confirm* matches (hence shrinkage on positive), but a single mismatch should prevent them (full weight on negative).

```
pos_agg = 1 - exp(-λ × pos_strength)
neg_agg = 1 - exp(-λ × neg_strength)

seed = name_similarity(a, b)
weight = n_with_counterpart / (n_with_counterpart + κ)
evidence = pos_agg × (1 - seed) × weight - neg_agg × seed
computed_dir = seed + evidence
```

The seed serves as the baseline that propagation anchors to. Positive evidence pushes toward 1.0 (proportional to the room above seed, discounted by shrinkage), negative evidence pushes toward 0.0 (proportional to the seed itself, at full strength). With no structural evidence the score equals the seed — but merging additionally requires `n_tested > 0`, so name similarity alone never triggers a merge.

The final score averages the two directional scores:

```
computed = (computed_fwd + computed_bwd) / 2
```

### The 0.5 threshold as a natural gate

The threshold for contributing positive vs negative evidence is 0.5 — the point of maximum uncertainty. A neighbor pair with confidence 0.6 contributes weak positive evidence. One with confidence 0.1 contributes strong negative evidence. One at exactly 0.5 contributes nothing.

This replaces the separate "gate" mechanism from the dual-channel design. There is no need for a separate activation threshold — the 0.5 boundary naturally ensures that negative evidence only affects pairs whose neighbors have meaningful non-match signal.

### Self-correcting dynamics

Unlike PARIS's one-shot negative factor, our approach is iterative and self-correcting. Consider two entities whose CEO neighbors initially have low name similarity (0.35). In early iterations, `1 - 0.35 = 0.65 > 0.5`, so the CEO pair generates negative evidence for the parent entities. But if the CEO pair has its own structural evidence (e.g. both graduated from the same university), its confidence rises across iterations. Once it crosses 0.5, it switches from generating negative evidence to generating positive evidence. The damped iteration converges to a consistent assignment.

This dynamic is impossible with the dual-channel monotone approach, where negative evidence is fixed at seed values to prevent circular reinforcement. Damped iteration allows circular reinforcement, bounded by the contraction property — feedback loops shrink geometrically rather than exploding.

### Convergence

The damped update `new = (1-α) × old + α × computed` ensures convergence for sparse graphs (see [similarity_flooding.md](similarity_flooding.md) for the full convergence analysis). Negative evidence does not require special treatment — it is part of the same contraction mapping. Each iteration brings the score vector closer to the unique fixed point regardless of whether individual scores go up or down.

### What negative evidence does NOT replace

Negative evidence helps distinguish entities that share some structure but differ on specific relations. It does not help with:

- **Completely disjoint graphs**: entities with no shared relation clusters generate neither positive nor negative structural evidence — the score falls back to the name-similarity seed, but the `n_tested > 0` merge gate prevents merging
- **Name-only matches**: if two entities match purely on name similarity with no structural support, the score equals the seed but cannot trigger a merge

## Relation to the broader pipeline

Negative evidence interacts with several other components:

- **Functionality**: negative evidence is weighted by forward functionality, so getting functionality estimates right is critical. Over-estimated functionality produces over-aggressive negative signals.
- **Relation similarity threshold**: a lower threshold means more relation pairs are considered "similar," which reduces false negatives in the inner match check. This indirectly weakens negative evidence (more neighbors appear to match).
- **Progressive merging**: if we merge confident pairs during propagation (see [progressive_merging.md](progressive_merging.md)), merged entities have richer neighborhoods, which improves both positive and negative evidence quality.

## References

- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* VLDB 2011. Section 4 (Equations 4-7), Section 6.3 (experimental evaluation of negative evidence).
- Peng, Bonald, Suchanek. *FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.* 2025. Definition 1 (no-negation constraint), Theorem 1 (convergence requires monotonicity).
- Lizorkin, Velikhov, Grinev, Turdakov. *Accuracy Estimate and Optimization Techniques for SimRank Computation.* PVLDB 2008. (Contraction convergence proof for iterative graph similarity with decay factor.)
