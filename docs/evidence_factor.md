# Evidence Factor

The evidence factor is a discount applied to structural similarity scores based on how many independent paths support the match. It prevents a single strong structural path from saturating evidence and overriding name dissimilarity.

The idea comes from SimRank++ (Antonellis et al. 2008), which introduces an "evidence weight" to penalize similarity scores supported by few common neighbors.

## The problem it solves

PARIS-style noisy-OR computes structural evidence as:

```
structural = 1 - PRODUCT(1 - func_w * nbr_conf)
```

When even one path has high `func_w * nbr_conf` (say 0.95), the structural score jumps to 0.95 regardless of whether that's the only evidence or one of many corroborating paths. Combined with the name seed via another noisy-OR layer, this single path can push the final confidence above the match threshold — even when the entities have dissimilar names.

Concrete example: Dr. Priya Sharma and Dr. Elena Vasquez both `founded` NovaTech Labs. NovaTech matches itself perfectly (confidence ~1.0), and `founded` is a functional relation (inverse functionality ~1.0). A single path through NovaTech gives structural evidence of ~1.0, which overrides the low name similarity between Sharma and Vasquez and produces a spurious merge.

The fundamental issue: noisy-OR measures the *strength* of the best evidence but ignores the *quantity* of evidence. One strong path and ten strong paths produce nearly the same score. In entity alignment, a single shared neighbor is weak circumstantial evidence — many shared neighbors with matching relations is strong corroboration.

## The fix

After computing the noisy-OR structural score, multiply by an evidence factor:

```
evidence = 1 - exp(-lambda * n_paths)
structural *= evidence
```

where `n_paths` is the count of qualifying paths (those that pass both the relation gate and the confidence gate) and `lambda` controls the discount curve.

## Why exponential saturation

The function `1 - exp(-lambda * n)` has properties that match our intuition about evidence accumulation:

- **Zero paths, zero evidence.** `f(0) = 0`. No structural paths means no structural evidence.
- **Diminishing returns.** Each additional path adds less. Going from 1 to 2 paths matters a lot; going from 9 to 10 barely changes the score.
- **Asymptotic to 1.** Many paths approach full credit but never exceed it.
- **Single parameter.** `lambda` controls how quickly evidence accumulates — how many paths you need before trusting the structural score.

## Choosing lambda

With `lambda = 0.5`:

| n_paths | evidence factor |
|---------|----------------|
| 1       | 0.39           |
| 2       | 0.63           |
| 3       | 0.78           |
| 5       | 0.92           |
| 10      | 0.99           |

One path gives ~40% credit. Three paths give ~78%. Five or more paths give 90%+ credit.

This matches the entity alignment setting: a single shared neighbor is ambiguous (many unrelated entities share a neighbor), but several independently matching neighbors via matching relations is strong structural corroboration. The Sharma/Vasquez case has exactly one path, so structural evidence is discounted to ~39% of its noisy-OR value — not enough to override name dissimilarity.

## Interaction with noisy-OR

The evidence factor and noisy-OR serve complementary roles:

- **Noisy-OR** measures the aggregate *strength* of all qualifying paths — strong paths contribute more than weak ones, and multiple paths compound.
- **Evidence factor** measures the *breadth* of support — how many independent paths corroborate the match.

The final structural score is their product: `strength * breadth`. A match needs both strong individual paths and enough of them.

## Convergence

The evidence factor preserves the monotonicity required for Knaster-Tarski convergence:

1. As confidence values increase across iterations, more neighbor pairs pass the confidence gate, so `n_paths` is non-decreasing.
2. `1 - exp(-lambda * n)` is non-decreasing in `n`.
3. The noisy-OR value is non-decreasing (more/stronger paths only increase it).
4. The product of two non-decreasing non-negative values is non-decreasing.

Therefore the overall confidence update remains monotonically non-decreasing and bounded in [0, 1], preserving the fixpoint guarantee.

## SimRank++ context

SimRank (Jeh and Widom 2002) measures node similarity by recursive descent: two nodes are similar if their neighbors are similar. SimRank++ extends this with an evidence weight that discounts similarity when two nodes share few common neighbors relative to their total neighbor counts. The intuition is the same — structural similarity supported by sparse overlap is unreliable.

Our adaptation differs in form (exponential saturation vs. Jaccard-style overlap) because we operate in a different setting: we're counting qualifying propagation paths across a bipartite entity-pair space rather than counting shared neighbors in a single graph. The exponential form is simpler and sufficient for our purpose of preventing single-path saturation.

## References

- Antonellis, Garcia-Molina, Chang. *SimRank++: Query Rewriting through Link Analysis of the Click Graph.* VLDB 2008.
- Jeh, Widom. *SimRank: A Measure of Structural-Context Similarity.* KDD 2002.
