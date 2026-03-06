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

## Our approach: dampened negative evidence

We need negative evidence but cannot afford PARIS's brittleness. The key insight is that negative evidence should be **weaker and more selective** than positive evidence, reflecting the fundamental asymmetry in our setting:

- A match between neighbors is *reliable* positive evidence (two articles independently reporting the same fact)
- A *missing* match could mean many things (incomplete coverage, relation phrasing mismatch, extraction error)

### Dampened negative factor

For each entity pair `(a, b)`, compute a negative factor:

```
neg(a, b) = PRODUCT_{edge r(a, y)} max(
    1 - alpha × fun(r) × PRODUCT_{edge r'(b, y')} (1 - Pr(y ≡ y')),
    floor
)
```

Where:
- `alpha < 1` is a dampening coefficient (e.g. 0.3) that weakens the negative signal relative to PARIS's full-strength version
- `floor` (e.g. 0.5) prevents any single missing match from killing the score entirely
- `fun(r)` is forward functionality — only functional relations generate negative evidence
- The inner product checks whether `y` matches *any* of `b`'s neighbors via similar relations

The dampening addresses the incompleteness problem: even with high forward functionality and no matching target, the penalty is at most `(1 - alpha × 1.0)` per path, clamped to `floor`.

### When to apply

Negative evidence should activate only when there is already positive evidence to temper. If a pair has near-zero positive similarity, negative evidence is irrelevant. Apply as:

```
final(a, b) = positive(a, b) × neg(a, b)    if positive(a, b) > gate
             positive(a, b)                   otherwise
```

The gate (e.g. 0.3) ensures negative evidence only modulates pairs that are already plausible matches. This prevents wasting computation on the vast majority of pairs that will never match.

### Convergence implications

Multiplying by a negative factor makes the update non-monotone — a pair's score can decrease between iterations. This breaks FLORA's convergence guarantee. Two mitigations:

1. **Apply negative evidence only at the end.** Run positive-only propagation to convergence (guaranteed by monotonicity), then apply negative factors as post-processing. Simple and safe, but misses the opportunity for negative evidence to prevent cascading false matches during propagation.

2. **Apply per-iteration but with a ratchet.** Allow scores to decrease, but never below `max(name_sim, score × (1 - max_decrease))`. This bounds the per-iteration decrease and prevents oscillation. Convergence is not formally guaranteed but is practically achievable with a decreasing `max_decrease` schedule.

Option 1 is the conservative starting point. Option 2 is the target if post-processing proves insufficient.

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
