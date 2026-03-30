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

Positive and negative evidence are computed together in each propagation step, feeding into a single score per entity pair. For each pair `(a, b)`, we examine all neighbor pairs `(y, y')` connected via similar relations:

- **Positive**: if the neighbor pair's confidence is above 0.5 (likely match), it contributes to `pos_strength`, weighted by inverse functionality — matching neighbors of a functional relation are strong evidence FOR the match.
- **Negative**: if the neighbor pair's confidence is below 0.5 (likely non-match), it contributes to `neg_strength`, weighted by forward functionality — **but only if neither neighbor has a better counterpart** (see "Best-counterpart gating" below).

Both are aggregated via exp-sum and combined with the name-similarity seed:

```
pos_agg = 1 - exp(-λ × pos_strength)
neg_agg = 1 - exp(-λ × neg_strength)

seed = name_similarity(a, b)
computed = seed + pos_agg × (1 - seed) - neg_agg × seed
```

The seed serves as the baseline. Positive evidence pushes toward 1.0 (proportional to the room above seed), negative evidence pushes toward 0.0 (proportional to the seed itself). With no structural evidence, the score equals the seed. With strong negative evidence and no positive evidence, the score approaches zero.

### Best-counterpart gating

A naive all-pairs approach to negative evidence generates a contribution for every low-confidence `(y, y')` cross-pair. This causes a cascade problem (GH issue #30): when entity A has neighbors X and Y connected via relation-similar edges, the cross-pair X↔Y' (which is legitimately low-confidence because X and Y are different entities) generates negative evidence even though X has a high-confidence counterpart X'. The negative evidence drags A's confidence below 0.5, and A then becomes negative evidence for its own neighbors — a self-reinforcing cascade that suppresses all same-name merges in the connected component.

The fix: before accumulating negative evidence, a first pass finds the best counterpart confidence for each neighbor. A low-confidence pair `(y, y')` only contributes negative evidence if **both** `y` and `y'` lack a better alternative (best counterpart confidence ≤ 0.5). If either neighbor has a good match elsewhere, the low-confidence cross-pair is irrelevant — it reflects expected structural non-correspondence, not evidence of non-identity.

This preserves negative evidence where it matters (a neighbor with NO good counterpart) while preventing cascade through well-matched neighbors.

### Merged-neighbor deduplication

After progressive merging, a single merged neighbor may appear in the canonical adjacency via multiple relation-similar edge entries (e.g. "is CEO of" and "is outgoing CEO of" — different strings, but embedding-similar). The propagation inner loop would count each entry independently, inflating positive evidence from the `ra == rb` (already-merged) case. Since a merged neighbor is a single structural fact regardless of how many surface-form edges survived adjacency dedup, the loop deduplicates by canonical neighbor ID: each merged neighbor contributes at most one positive-evidence entry.

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

- **Completely disjoint graphs**: entities with no shared neighbors generate neither positive nor negative structural evidence
- **Name-only matches**: if two entities match purely on name similarity with no structural support, negative evidence from unmatched neighbors could incorrectly suppress valid matches (the articles may simply cover different aspects of the entity)

## Relation to the broader pipeline

Negative evidence interacts with several other components:

- **Functionality**: negative evidence is weighted by forward functionality, so getting functionality estimates right is critical. Over-estimated functionality produces over-aggressive negative signals.
- **Relation similarity threshold**: a lower threshold means more relation pairs are considered "similar," which reduces false negatives in the inner match check. This indirectly weakens negative evidence (more neighbors appear to match).
- **Progressive merging**: if we merge confident pairs during propagation (see [progressive_merging.md](progressive_merging.md)), merged entities have richer neighborhoods, which improves both positive and negative evidence quality.

## References

- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* VLDB 2011. Section 4 (Equations 4-7), Section 6.3 (experimental evaluation of negative evidence).
- Peng, Bonald, Suchanek. *FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.* 2025. Definition 1 (no-negation constraint), Theorem 1 (convergence requires monotonicity).
- Lizorkin, Velikhov, Grinev, Turdakov. *Accuracy Estimate and Optimization Techniques for SimRank Computation.* PVLDB 2008. (Contraction convergence proof for iterative graph similarity with decay factor.)
