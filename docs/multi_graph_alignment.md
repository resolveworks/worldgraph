# Multi-Graph Alignment

All papers in our reference literature (PARIS, FLORA, Melnik SF) operate on exactly two graphs. We need to align N article graphs simultaneously. This document covers the approaches from IsoRank/IsoRankN and how our current unified-graph approach relates to them.

## The problem

Given N article graphs, we need to find which entities across all graphs refer to the same real-world entity. The naive approach is pairwise: run alignment on every pair of graphs, then merge results transitively. This has two problems:

1. **Inconsistency**: pairwise decisions can conflict. A might match B, B might match C, but A might not match C when compared directly.
2. **Information loss**: the pairwise comparison of A and C doesn't benefit from the fact that B connects them. Evidence is fragmented.

## IsoRank: pairwise alignment as an eigenvalue problem

Singh (2007) formulates network alignment as finding the principal eigenvector of the product graph. For two networks G1 and G2, the score R_ij for pairing node i (from G1) with node j (from G2) satisfies:

```
R_ij = Σ_{u ∈ N(i)} Σ_{v ∈ N(j)} [1 / (|N(u)| × |N(v)|)] × R_uv
```

In matrix form: `R = A × R`, where A is a stochastic matrix over the product graph. R is the stationary distribution of a random walk on this product graph — the same mathematical object as PageRank, applied to alignment.

Node attributes (name similarity in our case) enter as a convex combination:

```
R = α × A × R + (1 - α) × E
```

This maps directly onto our architecture: A is structural propagation, E is name similarity, α controls the blend. Our iterative propagation computes this eigenvector.

## IsoRankN: N-way alignment via spectral clustering

Liao et al. (2009) extend IsoRank to N graphs. The key idea: instead of making hard pairwise alignment decisions and merging them, build a single k-partite similarity graph from all pairwise scores and cluster it spectrally.

### The algorithm

1. Run pairwise IsoRank on every pair of N graphs → a weighted k-partite similarity graph
2. For each entity v, define its **star**: all cross-graph entities with similarity above threshold β
3. Spectrally partition each star using Personalized PageRank to find the tightest cluster (minimum conductance subset)
4. Merge overlapping clusters when their members mutually regard each other as high-weight neighbors

### Star spread

The star of entity v is its set of high-similarity candidates across all other graphs. The spectral partitioning finds S_v* — the subset with minimum conductance:

```
S_v* = min_j Φ(T_j^p)
```

where T_j^p are sets ordered by their Personalized PageRank mass, and Φ is conductance (ratio of cut edges to internal edges). Low conductance means a dense, well-separated cluster — a set of mentions that strongly match each other and weakly match everything else.

This handles dangling entities gracefully: entities with no good matches have high-conductance stars that don't yield good clusters.

## Our approach: unified graph

We don't run pairwise alignment. Instead, we merge all N article graphs into a single unified graph and run propagation once over all cross-graph entity pairs simultaneously. Final matches are produced by thresholding and union-find.

This is closer to IsoRankN's spirit than to pairwise-then-merge:
- All evidence is considered simultaneously during propagation
- Transitivity is implicit: if A↔B scores highly, that evidence propagates to help B↔C and A↔C in the same iteration
- No pairwise decisions are made until the final threshold

The difference from IsoRankN: we use union-find (transitive closure) instead of spectral clustering for the final grouping. Union-find is simpler but less principled — it cannot distinguish "A matches B and B matches C, but A doesn't match C" (a case where spectral clustering would separate them).

## When union-find fails

Union-find enforces transitivity: if A↔B and B↔C are above threshold, all three merge regardless of the A↔C score. This can cascade:

- Entity B has a common name that matches both A (correct) and C (incorrect, different entity)
- A and C merge through B despite having low direct similarity
- The merged entity now has a polluted neighborhood that can trigger further false merges

Spectral clustering (IsoRankN-style) avoids this by finding the tightest cluster: if A↔C similarity is low, the minimum-conductance cut would separate them even if both connect to B.

## Possible improvement: confidence-weighted union-find

A middle ground between naive union-find and full spectral clustering: weight the union-find edges by confidence and require that every entity in a group has above-threshold similarity to the group centroid (or to the majority of group members), not just to one member.

This is simpler than spectral clustering but catches the worst transitivity failures. It can be implemented as a post-processing validation step on union-find groups.

## Interaction with progressive merging

[Progressive merging](progressive_merging.md) creates a natural N-way alignment mechanism: when entities from graphs A and B merge during propagation, the merged entity has edges from both graphs. This enriched entity is then compared against entities from graph C with more structural evidence than either A or B alone would provide.

This is an implicit form of multi-graph alignment that operates during propagation rather than as a post-processing step. It's more powerful than pairwise-then-merge because evidence accumulates across graphs incrementally.

## References

- Singh, Xu, Berger. *Global Alignment of Multiple Protein Interaction Networks with Application to Functional Orthology Detection.* PNAS 2008. (IsoRank — pairwise alignment as eigenvector computation.)
- Liao, Sabetiansfahani, Bhatt, Ben-Hur. *IsoRankN: Spectral Methods for Global Alignment of Multiple Protein Networks.* Bioinformatics 2009. (N-way alignment via spectral clustering of pairwise scores.)
