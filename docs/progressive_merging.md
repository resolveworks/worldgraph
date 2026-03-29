# Progressive Merging

Standard similarity propagation treats all entity pairs as soft hypotheses until the very end, when a threshold produces hard merge decisions. This document proposes an alternative: committing to high-confidence merges during propagation and using the merged entities as stronger evidence for subsequent matches.

## The problem

In the current pipeline, propagation runs to convergence on the full set of entity pairs, then union-find produces match groups. This means:

1. An entity that appears in 5 articles contributes evidence only through pairwise similarities. If articles A and B match, and B and C match, A and C benefit from B's neighborhood — but only softly, through propagated scores. They don't benefit from the *combined* neighborhood of the merged A+B entity.

2. A confident match (score 0.95 at iteration 3) provides the same propagation strength as a borderline match (score 0.55 at iteration 3). Both are just numbers in the confidence dict.

3. Merging happens in a separate post-processing step (union-find), disconnected from the propagation that produced the scores. Evidence from transitivity is entirely implicit.

Progressive merging addresses this by periodically committing high-confidence matches during propagation, combining their neighborhoods, and continuing with the enriched graph.

## What the literature says

### PARIS

PARIS uses a "maximal assignment" optimization (Section 5.2): each iteration considers only the top-scoring match per entity from the previous round, discarding weaker candidates. This is a form of soft progressive commitment — confident matches dominate propagation in subsequent rounds. However, no actual merging occurs; the entities remain separate throughout.

The authors tested using all probabilities instead of just maximal assignments and found it "changed the results only marginally (by one correctly matched entity), because the first iteration already has a very good precision." This suggests that for curated knowledge bases, progressive merging may not matter much. For noisy, sparse article graphs, the situation is different.

### FLORA

FLORA keeps everything soft until the end. Variables can only increase (the monotonicity required for convergence), and the maximum assignment is strictly post-processing (Equation 3). There is no mechanism for committing to matches during propagation.

However, FLORA does achieve implicit bootstrapping: a high-confidence match at iteration 3 naturally propagates more strongly to neighbors at iteration 4, because the neighbor's premises have higher input values. This is the standard fixpoint behavior — but it operates on individual pair scores, not on merged neighborhoods.

### IsoRankN

IsoRankN (Liao et al. 2009) takes a different approach entirely: instead of iterative propagation with progressive commitment, it computes all pairwise scores first, builds a k-partite similarity graph, and uses spectral clustering (Personalized PageRank + minimum conductance) to find match groups. The "star spread" concept — finding the tightest cluster around each entity — is a principled alternative to progressive merging: instead of committing early, you gather all evidence and cluster once.

This is relevant because it suggests two fundamentally different strategies:
1. **Progressive merging**: commit during propagation, enrich neighborhoods, continue
2. **Gather-then-cluster**: propagate to convergence, then find match groups using global information

Strategy 2 avoids the convergence issues of progressive merging but misses the enriched-neighborhood benefit.

## Our approach: progressive merging within damped iteration

We use a single-loop design where propagation runs with damped updates (positive and negative evidence integrated into each step), and merges are committed when the iteration converges. Merged neighborhoods then compound structural evidence in subsequent iterations.

### The mechanism

```
for iteration in range(max_iter):
    # Damped propagation step (positive + negative in one pass)
    for each pair (a, b):
        computed = seed + pos_agg(neighbors) × (1 - seed) - neg_agg(neighbors) × seed
        confidence(a, b) = (1 - α) × old + α × computed
    if not converged:
        continue

    # Commit high-confidence merges
    new_merges = find_merges(confidence, threshold=merge_threshold)
    if not new_merges:
        break  # Converged, no new merges → done

    # Update adjacency incrementally, remap pairs/confidence
    for a, b in new_merges:
        uf.union(a, b)
        canonical_adj[uf.find(a)] = dedup(adj[a] + adj[b])
    confidence, pairs = remap_to_canonical(confidence, pairs, uf)
```

Key properties:
- **No separate phases**: positive and negative evidence are computed together in each step, not sequentially. See [negative_evidence.md](negative_evidence.md) for how this works.
- **No reseeding**: the damped update naturally anchors to the seed — there is no compounding dampening effect that requires periodic reseeding.
- **Incremental adjacency**: maintaining a `canonical_adj` alongside the UnionFind avoids rebuilding adjacency from scratch each cycle. Each merge costs O(degree), not O(|edges|).

### What merging means concretely

When entities `a` and `b` are merged into entity `ab`:
1. **Union their neighborhoods**: `ab` inherits all edges from both `a` and `b`. If `a` was connected to `c` via "acquired" and `b` was connected to `d` via "purchased", `ab` now has both edges.
2. **Combine their names**: the canonical name is chosen by the name-selection heuristic (e.g. longest name, or highest IDF-weighted name). Both names remain available for name-similarity seeding.
3. **Update confidence**: for any entity `x`, `confidence(ab, x) = max(confidence(a, x), confidence(b, x))`. Existing matches are preserved.
4. **Update functionality**: edge statistics change when entities merge (fewer distinct sources/targets for a given relation), so functionality should be recomputed. In practice, the change is small for low-frequency merges.

### The enriched-neighborhood effect

This is the core benefit. Consider three articles about the same acquisition:

- Article A: `[Meridian Tech] —acquired→ [DataVault]`
- Article B: `[Meridian Technologies] —bought→ [DataVault Inc]` and `[Meridian Technologies] —headquartered in→ [Austin]`
- Article C: `[Meridian] —headquartered in→ [Austin, TX]`

After epoch 1, A's "Meridian Tech" and B's "Meridian Technologies" merge (strong name + structural match via the acquisition edge). The merged entity now has edges to both "DataVault"/"DataVault Inc" AND "Austin".

In epoch 2, propagation between this merged entity and C's "Meridian" benefits from TWO structural paths: the headquarters edge AND the combined acquisition evidence. Without progressive merging, C's "Meridian" would only see one structural path per pairwise comparison.

### Merge threshold vs. match threshold

Two distinct thresholds:

- **Merge threshold** (high, e.g. 0.9): used during propagation to commit only very confident matches. False merges are catastrophic — they combine neighborhoods of unrelated entities and cascade.
- **Match threshold** (lower, e.g. 0.7): used at the end to produce the final match groups. Borderline matches that didn't qualify for progressive merging may still be valid.

The merge threshold should be conservative. A false merge during propagation is far worse than a false merge in post-processing, because it pollutes the evidence for all subsequent iterations.

### Convergence properties

Within each convergence cycle, the damped iteration converges via the contraction mapping property (see [similarity_flooding.md](similarity_flooding.md)). Between cycles, merging changes the graph structure, so the overall process is not a single contraction mapping.

However, the process is well-behaved:
1. **Merges are irreversible**: once committed, entities stay merged. The set of merged entities grows monotonically.
2. **The graph shrinks**: each merge reduces the entity count by one. The process must terminate in at most N-1 merge steps.
3. **Within-cycle convergence**: each cycle converges to the unique fixed point of the current graph's contraction mapping. The fixed point changes when merges alter the graph structure.
4. **Termination**: if no cycle produces new merges, the process halts.

The conservative merge threshold (0.9) limits cascade risk: only very high-confidence pairs are merged, and enriched neighborhoods from those merges are unlikely to create false matches above the same threshold.

### Interaction with negative evidence

Progressive merging and [negative evidence](negative_evidence.md) interact in two ways:

1. **Enriched neighborhoods improve negative evidence quality.** After merging, a combined entity has more edges, which means more opportunities for both positive AND negative evidence. A false match candidate that survived against sparse individual neighborhoods may fail against the richer merged neighborhood.

2. **Negative evidence prevents false progressive merges.** Because negative evidence is integrated into each propagation step (not applied post-hoc), it suppresses pairs with contradictory functional relations before they ever reach the merge threshold. This is a natural safety mechanism against cascade risk.

## What progressive merging does NOT solve

- **Cross-cluster confusion**: if two unrelated event clusters have similar structure, progressive merging within one cluster doesn't help distinguish it from the other. That requires negative evidence or name distinctiveness.
- **Initial seeding**: progressive merging amplifies existing signal. If the initial name similarity + first-pass structural evidence doesn't identify any matches, there's nothing to merge and amplify.
- **Computational cost**: each epoch requires re-running propagation on the (slightly smaller) graph. For large graphs, this is expensive. The cost is bounded by the number of merge epochs (typically 3-5 for news data), not the number of entity pairs.

## References

- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* VLDB 2011. Section 5.2 (maximal assignment as soft progressive commitment).
- Peng, Bonald, Suchanek. *FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.* 2025. Theorem 1 (convergence requires monotonicity — contrast with our contraction-based approach).
- Liao, Sabetiansfahani, Bhatt, Ben-Hur. *IsoRankN: Spectral Methods for Global Alignment of Multiple Protein Networks.* Bioinformatics 2009. Sections 2.2-2.4 (star spread as alternative to progressive merging).
