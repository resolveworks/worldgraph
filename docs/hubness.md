# Hubness and Literal Similarity

Hubness is a geometric property of high-dimensional spaces that inflates similarity scores between unrelated entities. This document explains the problem, the prior art for correcting it, and our solution: Mutual Proximity with local neighborhood statistics, which produces calibrated [0, 1] confidence scores for entity name matching.

## The hubness problem

In high-dimensional spaces, the distribution of nearest-neighbor relationships becomes skewed (Radovanović et al., 2010). Some vectors — **hubs** — appear among the nearest neighbors of many other vectors, while others (**anti-hubs**) appear in no one's neighbor list.

The root cause is **distance concentration**: as dimensionality grows, all pairwise distances converge toward the same value. But they don't converge at the same rate for all points. Points closer to the data mean concentrate faster, making them appear generically "close to everything." These become hubs — not because they're genuinely similar to many things, but because the distance metric can no longer distinguish them.

This is an inherent property of the geometry, not a defect of any particular model or dataset. It has been observed in music retrieval, image search, text embeddings, and bioinformatics.

## Why it matters for entity name matching

Our entity literals include person names, organization names, place names, acronyms, and descriptive phrases. When embedded with a general-purpose sentence model, entities of the same *type* cluster together. The model learns that "Dr. Priya Sharma" and "Dr. Elena Vasquez" are both doctor names, that "Alpha Corp" and "Beta Inc" are both companies.

This produces inflated cosine similarities between same-type entities:

| Pair | Cosine | Same entity? |
|------|--------|--------------|
| "Dr. Elena Vasquez" vs "Dr. Lena Vasquez" | 0.97 | Possibly (one character difference) |
| "Alpha Corp" vs "Beta Inc" | 0.87 | No |
| "Dr. Priya Sharma" vs "Dr. Elena Vasquez" | 0.84 | No |
| "FTC" vs "Federal Trade Commission" | 0.83 | Yes (acronym) |

The critical pair: "FTC" ↔ "Federal Trade Commission" (genuine match, cosine 0.83) is *lower* than "Dr. Priya Sharma" ↔ "Dr. Elena Vasquez" (false match, cosine 0.84). Raw cosine cannot separate them. A fixed threshold tuned to catch one will either miss genuine matches or admit false ones, and switching embedding models shifts the entire distribution, breaking any hardcoded threshold.

Person names are hubs with respect to other person names. Company names are hubs with respect to other company names. The solution must discount these type-level similarities to expose the entity-level signal.

## Prior art: CSLS

Cross-domain Similarity Local Scaling (Conneau et al., 2018) was introduced for cross-lingual word translation. For each point, compute its mean similarity to its K nearest neighbors (its "hub score"), then subtract:

```
CSLS(x, y) = 2 × cos(x, y) - mean_knn(x) - mean_knn(y)
```

This penalizes hubs (high mean_knn → all their scores decrease) and boosts anti-hubs. It works well as a **ranking metric** — on the entity pairs above, CSLS correctly ranks the genuine FTC match (+0.19) above the false Sharma/Vasquez match (-0.01).

But CSLS was designed for nearest-neighbor retrieval, not for producing calibrated scores. Its output is unbounded (roughly [-1, +2]) and concentrates in a narrow band that doesn't align with any meaningful threshold. In our pipeline, literal similarity feeds into propagation as a confidence score in [0, 1], compared against a confidence gate. CSLS values can't fill that role without an arbitrary rescaling that reintroduces the model-dependence we're trying to eliminate.

## Our solution: Mutual Proximity with local neighborhoods

Mutual Proximity (Schnitzer et al., 2012) takes a different approach to hubness reduction: instead of subtracting hub scores, it transforms similarities into **probabilities**. The question changes from "how similar are x and y?" to "how unusually similar are x and y, given what we know about each of their neighborhoods?"

### The core idea

For each entity x, model the distribution of its similarities to all other entities. Given that distribution, the similarity `cos(x, y)` can be interpreted as a percentile: what fraction of the population is *less* similar to x than y is?

If x is a hub (similar to everything), even a moderately high cosine score places y at a low percentile — most entities score similarly. If x is an anti-hub (similar to very few things), the same cosine score places y at a high percentile — y is unusually close.

Mutual Proximity computes this percentile from both sides and multiplies:

```
MP(x, y) = P(S_x < cos(x, y)) × P(S_y < cos(x, y))
```

where S_x is the random variable representing cosine similarity of x to a randomly chosen other entity. The product enforces mutual agreement: both x and y must find each other unusually similar.

### The Gaussian approximation

Computing exact percentiles requires the full pairwise similarity matrix (O(n²) space). Schnitzer et al. show that a Gaussian approximation works well in practice: model each entity's similarity distribution as N(μ_x, σ_x), estimated from sample statistics. Then:

```
MP(x, y) = Φ((cos(x, y) - μ_x) / σ_x) × Φ((cos(x, y) - μ_y) / σ_y)
```

where Φ is the standard normal CDF. This reduces computation to O(n) per entity (compute mean and std of its similarities) plus O(1) per pair lookup.

### Local vs. global statistics

Schnitzer et al. compute μ and σ over all pairwise similarities (global statistics). This works when the population is homogeneous, but in our case entity types form distinct clusters. The global mean similarity (~0.56) is too low a reference point — same-type pairs at 0.84 look exceptional against the global distribution even when they shouldn't.

CSLS avoids this by using the K nearest neighbors as the reference distribution. We apply the same idea to Mutual Proximity: compute μ_x and σ_x from x's K nearest neighbors only, not the full population.

```
μ_x = mean of {cos(x, y) : y ∈ KNN_K(x)}
σ_x = std  of {cos(x, y) : y ∈ KNN_K(x)}
```

This makes the reference distribution *local*: a person name is compared against other person names (its nearest neighbors), not against the global entity population. The result is that same-type similarity (high cosine, but normal for the local neighborhood) gets a low MP score, while genuine matches (cosine significantly above the local baseline) get a high MP score.

### Results on our entity data

Using K=10 with 219 entity literals:

| Pair | Cosine | MP (K=10) | Same entity? |
|------|--------|-----------|--------------|
| FTC / Federal Trade Commission | 0.83 | **0.96** | Yes |
| EU / European Union | 0.94 | **0.92** | Yes |
| Meridian Technologies / Meridian Tech | 0.98 | **0.97** | Yes |
| Dr. Priya Sharma / P. Sharma | 0.87 | **0.64** | Yes |
| Commissioner A. Torres / Angela Torres | 0.79 | 0.01 | Yes |
| Dr. Priya Sharma / Dr. Elena Vasquez | 0.84 | **0.19** | No |
| FTC / FDA | 0.66 | **0.01** | No |
| Volta Systems / Halcyon Genomics | 0.65 | **0.00** | No |

The previously indistinguishable pair (FTC/Federal Trade Commission at 0.83 vs Sharma/Vasquez at 0.84) is now clearly separated (0.96 vs 0.19). False matches collapse to near zero. A confidence gate of 0.5 cleanly separates genuine matches from false ones.

The trade-off: some genuine matches with weak embedding signal fall below threshold ("Commissioner A. Torres" / "Angela Torres" at 0.01). These are cases where the embedding model doesn't place the pair far enough above the local baseline — the structural propagation must provide the evidence instead, which is the correct behavior. Name similarity alone shouldn't force a match when the embeddings are ambiguous.

### Properties

**Calibrated [0, 1] output.** The product of two CDF values is naturally bounded in [0, 1] with a probabilistic interpretation. No rescaling or clamping needed.

**Model-agnostic.** When the embedding model changes, μ and σ shift accordingly. A model that inflates all similarities inflates the local baseline equally, so MP scores remain stable.

**Single parameter.** K controls the locality of the reference distribution. K=10 works well; the results are stable across K=5 to K=50 (following CSLS findings). Smaller K gives more aggressive hub correction; larger K approaches global statistics.

## References

- Radovanović, Nanopoulos, Ivanović. *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* JMLR 2010. (Discovery and analysis of hubness as an inherent property of high-dimensional spaces.)
- Schnitzer, Flexer, Schedl, Widmer. *Local and Global Scaling Reduce Hubs in Space.* JMLR 2012. Section 3.2 (Mutual Proximity definition and Gaussian approximation), Section 3.2.2 (independence approximation).
- Conneau, Lample, Ranzato, Denoyer, Jégou. *Word Translation Without Parallel Data.* ICLR 2018. Section 2.3 (CSLS definition — the ranking metric we build on).
- Zelnik-Manor, Perona. *Self-Tuning Spectral Clustering.* NeurIPS 2005. (Local scaling, the precursor to both CSLS and local neighborhood statistics.)
