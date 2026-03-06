# Similarity Flooding and Embedding Equivalence

Sun et al. (2023) prove that embedding-based entity alignment models are mathematically equivalent to similarity flooding. The embedding learning process is an indirect way to find a fixpoint of pairwise entity similarities. This document summarizes the result and its implications for our approach.

## The core theorem

Entity representations in both TransE-based and GCN-based alignment models can be decomposed into weighted compositions of neighbor embeddings. For TransE (Eq. 5):

```
e = (1/|T_e|) × Σ_{(e,r,o) ∈ T_e} (o - (1/|T_r|) × Σ_{(s',r,o') ∈ T_r} (o' - s'))
```

The pairwise similarity matrix Ω between source and target entities satisfies:

```
Λ × Ω × (Λ')^T = Ω
```

where Λ and Λ' are composition coefficient matrices derived from the graph structure. This is a fixpoint equation — the same mathematical object that similarity flooding computes.

**Theorem 3.1**: "The TransE-based EA model seeks a fixpoint of pairwise entity similarities via embedding learning."

**Theorem 3.2**: The same holds for GCN-based models, with different lambda values.

## What the lambda values encode

The composition coefficients differ between model families:

**TransE** (Eq. 12): lambda depends on the number of shared relation types between entities, the degree distribution, and the relation frequency. It captures both direct connections and indirect structural patterns.

**GCN** (Eq. 13): `λ_{i,j} = 1_{(x_i,r,x_j) ∈ T} / |T_{x_i}|` — simply the normalized adjacency, identical to the propagation coefficients in standard similarity flooding with inverse-degree weighting.

The key insight: these lambda values play the same role as propagation coefficients in Melnik's original algorithm (inverse-product weights) or functionality in PARIS. Different embedding models implicitly choose different weighting schemes for propagation.

## Practical methods derived from the equivalence

### TransFlood / GCNFlood

Run the fixpoint iteration directly on the composition matrices without learning any embeddings:

```
Ω_t = normalize(Λ × Ω_{t-1} × (Λ')^T)
```

Results: TransFlood matches or beats TransE on entity alignment benchmarks (e.g. Hits@1 of 0.347 vs 0.244 on FR-EN) while running in comparable time without GPU training.

### Self-Propagation (SPA)

Add a skip connection to GCN layers to prevent over-smoothing:

```
e^{i+1} = (1 - α) × aggregate(neighbors) + α × f(e^i)
```

Without self-propagation, deep GCNs converge to a degenerate state where all entity representations become identical (over-smoothing). SPA preserves entity identity through layers, improving all four tested GCN models.

## Implications for our approach

### Validation

Our PARIS-style propagation is not a heuristic approximation of what embeddings do — it is the same computation, directly. The fixpoint we compute is the same fixpoint that a trained embedding model would converge to. This validates the algorithmic approach: we are not losing anything by using propagation instead of learned embeddings.

### No PCG needed

Standard similarity flooding requires building a Pairwise Connectivity Graph (PCG), which requires relation alignment. TransFlood/GCNFlood skip this by deriving entity compositions from structure alone. Our approach similarly avoids a formal PCG by using continuous relation phrase similarity to gate propagation paths — related but not identical to the composition-based approach.

### Propagation coefficients matter

The equivalence shows that the choice of propagation weights (our functionality weighting) corresponds to the choice of embedding model architecture. PARIS functionality and Melnik inverse-product weights are specific points in a space of possible weightings. The Sun et al. result suggests that as long as the weighting captures structural specificity (rare/functional relations count more), the exact formula matters less than getting the direction right.

## References

- Sun, Chen, Chakrabarti, Faloutsos. *What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings.* ICML 2023.
- Melnik, Garcia-Molina, Rahm. *Similarity Flooding: A Versatile Graph Matching Algorithm.* ICDE 2002.
- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* VLDB 2011.
