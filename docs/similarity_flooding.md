# Similarity Flooding

Similarity flooding is an iterative fixpoint algorithm for finding correspondences between nodes in two graphs. It was introduced by Melnik, Garcia-Molina, and Rahm (2002) for schema matching, then extended by PARIS (2011) and FLORA (2025) for knowledge graph alignment.

This document explains the core mechanism and its evolution across the three papers. For the role of functionality weighting specifically, see [functionality.md](functionality.md).

## The circularity problem

To decide whether entity `a` in graph G1 matches entity `b` in graph G2, we want to check: do their neighbors match? But to know if the neighbors match, we need to know if *their* neighbors match, and so on. Every entity-pair similarity depends on other entity-pair similarities.

Similarity flooding dissolves this circularity by computing all similarities simultaneously through iteration, starting from rough initial estimates and refining them until they stabilize.

## Melnik 2002: the original algorithm

### Step 1 — Represent everything as directed labeled graphs

The input structures (SQL schemas, XML trees, ontologies, or in our case entity-relation subgraphs) are converted to directed graphs where nodes represent elements and labeled edges represent relationships.

### Step 2 — Construct the pairwise connectivity graph

Given graphs A and B, construct a new graph whose nodes are **map pairs** — elements of A x B. Each map pair `(a, b)` represents the hypothesis "a in A corresponds to b in B."

Map pairs are connected when the underlying nodes share structural evidence. Specifically, `(a, b)` connects to `(a', b')` if there is an edge with label `l` from `a` to `a'` in A and an edge with the same label `l` from `b` to `b'` in B. The existence of this shared edge pattern means: if `(a, b)` is a good match, that's evidence for `(a', b')` being a good match too, and vice versa.

### Step 3 — Assign propagation coefficients

Each edge in the connectivity graph gets a weight (propagation coefficient) that controls how much similarity flows along it. The paper uses an **inverse-product** formula: count how many edges with label `l` leave each node in the pair, and divide 1.0 by the product of those counts.

If `a` has one `l`-edge out and `b` has two `l`-edges out, the coefficient on each of the two resulting connectivity edges is `1/(1*2) = 0.5`. The intuition: the more alternatives a node has, the less each individual connection contributes. A unique connection (only one `l`-edge from each side) gets the full weight of 1.0.

This is the precursor to PARIS's functionality concept — both measure how "specific" a relationship is.

### Step 4 — Iterate to fixpoint

Initialize similarities `sigma_0` from some seed (e.g. string similarity between node labels). Then iterate:

```
sigma^{i+1}(x, y) = sigma^i(x, y) + SUM over neighbors (sigma^i(neighbor) * weight)
normalize: divide all values by the current maximum
```

The paper tests four formula variants:

| Formula | Update rule |
|---------|------------|
| Basic | `normalize(sigma^i + phi(sigma^i))` |
| A | `normalize(sigma^0 + phi(sigma^i))` |
| B | `normalize(phi(sigma^0 + sigma^i))` |
| C | `normalize(sigma^0 + sigma^i + phi(sigma^0 + sigma^i))` |

where `phi` is the propagation function (sum of neighbor similarities times weights). Formula C converges fastest and produces the best matches in their experiments.

Key properties:
- Similarities are **normalized** each iteration (divided by the max), keeping values in [0, 1]
- Convergence is checked by the Euclidean distance of the residual vector between iterations
- The algorithm is relatively **insensitive to initial values** — even 100% random perturbation of seeds only degrades accuracy by ~15%

### Step 5 — Filter the multimapping

The raw fixpoint output gives a similarity score for every pair in A x B. This "multimapping" needs filtering to select actual matches.

The paper's best-performing filter uses **relative similarities**: for each element, normalize its match candidates by dividing by the score of its best candidate. Then apply the **stable marriage** property — keep a pair `(a, b)` only if `b` is `a`'s best match AND `a` is `b`'s best match (or within a relative threshold). This avoids situations where two elements prefer each other's partners.

Absolute similarity values don't matter much. What matters is whether a match is clearly the best option relative to alternatives.

## PARIS 2011: probabilistic grounding

PARIS (Probabilistic Alignment of Relations, Instances, and Schema) reformulates similarity flooding in a probabilistic framework for ontology alignment.

### From propagation coefficients to functionality

Where Melnik uses ad-hoc inverse-product weights, PARIS introduces [**functionality**](functionality.md) — a principled measure of how close a relation is to being a mathematical function. The probability that matched neighbors imply matched entities is weighted by how functional the connecting relation is. See the [functionality documentation](functionality.md) for the full definition and examples.

### The probabilistic formula

PARIS models entity equivalence `Pr(x ≡ x')` as:

```
Pr(x ≡ x') = 1 - PRODUCT over all evidence paths of (1 - fun_inv(r) * Pr(y ≡ y'))
```

where the product ranges over all pairs of facts `r(x, y)` and `r(x', y')`.

This is a **noisy-OR**: each evidence path independently has some chance of "succeeding" (establishing the match), and the overall probability is one minus the chance that all paths fail. Multiple independent paths **accumulate** — 5 weak pieces of evidence can establish a match that no single piece could.

This is fundamentally different from taking a single max. If entity pair (a, b) has 5 neighbor matches each contributing 0.3, the noisy-OR gives `1 - (1-0.3)^5 = 0.83`, while a max gives just 0.3.

A weakness of noisy-OR in practice is that a single high-confidence path can saturate the score, making one shared anchor indistinguishable from many. We address this by using exponential sum aggregation instead (`1 - exp(-λ × Σ strengths)`), which inherently rewards breadth — multiple paths accumulate proportionally rather than saturating early.

### Negative evidence

PARIS can optionally incorporate negative evidence (Equation 7): if a relation is highly functional and `r(x, y)` exists but `y` doesn't match any `y'` connected to `x'`, that's evidence *against* `x ≡ x'`. This can help distinguish entities that share some neighbors but differ on specific ones.

### Joint entity and relation alignment

PARIS doesn't just align entities — it simultaneously discovers which relations across ontologies are sub-relations of each other. The probability `Pr(r ⊆ r')` is estimated from how many facts in `r` have matching counterparts in `r'`, given the current entity alignments. Entity and relation alignments are updated alternately until convergence.

This is important because two knowledge graphs may use completely different relation names for the same concept (`wasBornIn` vs `birthPlace`). PARIS can discover these correspondences from structure alone.

### Maximum assignment

After convergence, PARIS keeps only the **maximum assignment** per entity: each entity `e` is matched to at most one `e'` (the highest-scoring candidate), and symmetrically. This enforces a clean one-to-one mapping.

## FLORA 2025: fuzzy logic + convergence proof

FLORA (Fuzzy-Logic based Object and Relation Alignment) is PARIS's successor. It reformulates the alignment problem as a recursive Fuzzy Inference System, fixing three issues with PARIS: no convergence guarantee, poor performance without functional relations, and rigid literal comparison.

### Fuzzy rules instead of probability

The entity alignment rule (Equation 1):

```
R(H, t) ∧ R'(H', t') ∧ H ≡ H' ∧ R ≅ R' ∧ fun(R) ∧ fun(R, H) ∧ fun(R') ∧ fun(R', H')
    --min-->  t ≡ t'
```

All premises are values in [0, 1]. The aggregation is `min`: the evidence for the match equals the **weakest premise**. A strong neighbor match cannot compensate for a non-functional relation. If multiple rules fire for the same output variable `t ≡ t'`, their strengths are combined with `max`.

This min/max structure is more conservative than PARIS's noisy-OR. PARIS can accumulate weak evidence into a strong match; FLORA requires every piece of evidence to be individually strong.

### Convergence guarantee

FLORA proves convergence via the Knaster-Tarski fixed point theorem:

1. All aggregation functions (`min`, `max`, harmonic mean) are continuous and non-decreasing
2. Scores only increase: `v(x) <- max(v(x), strength(rule))` — a variable never decreases
3. All values are bounded in [0, 1]
4. Therefore the sequence of value vectors is monotonically non-decreasing and bounded, so it converges to the **least fixed point**

This is stronger than Melnik's empirical convergence observation and eliminates PARIS's lack of theoretical guarantees.

### Relation lists

FLORA extends functionality to **relation lists** — ordered tuples of relations connecting the same head entities to a tail. For example, `BirthDateOf` alone is not very functional (many people share a birthday), and `FamilyNameOf` alone is not very functional (many people share a surname). But the combination `(BirthDateOf, FamilyNameOf)` is much more functional — few people share both birthday and family name.

### Both global and local functionality

FLORA's alignment rule requires four functionality terms: `fun(R)`, `fun(R, H)`, `fun(R')`, `fun(R', H')` — both global and local, for both sides. This prevents false matches when a globally functional relation has local exceptions (e.g. `hasCapital` is globally functional but not for South Africa). See [functionality.md](functionality.md) for more on local vs global functionality.

### Subrelation alignment

Like PARIS, FLORA jointly aligns relations. It discovers asymmetric sub-relation mappings (`r ⊆ r'`) by checking whether facts in `r` have corresponding facts in `r'` for all matched entity pairs, using an `alpha-mean` aggregation (arithmetic mean scaled by a "benefit of the doubt" factor `alpha` to handle incomplete knowledge graphs).

## Our approach: damped fixed-point iteration

We depart from FLORA's monotone framework. FLORA's Knaster-Tarski convergence proof requires all updates to be non-decreasing, which prevents negative evidence from being applied during iteration (scores can't go down). This forced our earlier design into a dual-channel architecture with separate positive and negative channels — complex and hard to reason about.

Instead, we use **damped fixed-point iteration**, which guarantees convergence while allowing scores to both increase and decrease.

### Banach vs Knaster-Tarski

FLORA's convergence relies on the **Knaster-Tarski fixed point theorem**: on a complete lattice, every monotone function has a fixed point. This requires all score updates to be non-decreasing. Scores can only go up, which is why FLORA explicitly excludes negation.

The **Banach fixed point theorem** (contraction mapping theorem) takes a different approach: on a complete metric space, every contraction mapping has a **unique** fixed point, and iterated application converges to it from any starting point. A contraction shrinks distances — applying the update to any two score vectors brings them closer together. Convergence is geometric: after k iterations, the error is bounded by `q^k × initial_error`, where `q < 1` is the contraction constant.

| Property | Knaster-Tarski (FLORA) | Banach (ours) |
|----------|----------------------|---------------|
| **Space** | Complete lattice | Complete metric space |
| **Requirement** | Monotone (non-decreasing) | Contraction (distance-shrinking) |
| **Fixed point** | Exists (least/greatest) | Exists and is **unique** |
| **Monotonicity** | Required | Not required |
| **Scores decrease?** | Never | Yes — negative evidence integrated |
| **Convergence rate** | Not bounded | Geometric: `q^k` |

### The damped update rule

Given a score update function `f` that computes a new score from neighbor confidences, the **damped update** is:

```
new_score = (1 - α) × old_score + α × f(old_scores)
```

where `α ∈ (0, 1)` is the damping factor. Each iteration blends the old score with the newly computed one. This has two effects:

1. **Smooths oscillation**: even if `f` would produce wild swings (e.g. negative evidence temporarily overcorrecting), the damping limits each step to a fraction `α` of the full change.
2. **Creates contraction**: the composed map `g(x) = (1-α)x + αf(x)` has Lipschitz constant `(1-α) + α·Lip(f)`. When `Lip(f) < 1` — which holds for sparse graphs with functionality-weighted evidence — the composed map is a contraction.

### Precedents

**SimRank** (Jeh & Widom, KDD 2002) is the canonical example of a contraction mapping for graph similarity. The SimRank equation `s(a,b) = C/(|I(a)|·|I(b)|) × Σ s(I_i(a), I_j(b))` uses a decay constant `C < 1` that serves as the contraction constant. Lizorkin et al. (PVLDB 2008) formally proved that the error after k iterations is bounded by `C^{k+1}` — exponential convergence with rate equal to the decay factor. This is a direct application of the Banach fixed point theorem.

**PageRank** (Brin & Page, 1998) uses a damping/teleportation factor `(1-d)` that serves the same role. The iteration `r = d·M·r + (1-d)·v` is a contraction with constant `d` (typically 0.85). The teleportation term anchors the iteration and prevents oscillation, guaranteeing convergence regardless of graph structure.

Our damped update follows the same pattern. The name-similarity seed acts as our "teleportation" anchor — it prevents scores from drifting arbitrarily and ensures the fixed point reflects both name and structural evidence.

### Our aggregation formula

For each entity pair `(a, b)`, we compute structural evidence from matching neighbor pairs:

```
seed = name_similarity(a, b)
computed = seed + pos_agg × (1 - seed) - neg_agg × seed
computed = clamp(computed, 0, 1)
new_score = (1 - α) × old_score + α × computed
```

Where:
- `pos_agg = 1 - exp(-λ × Σ pos_strengths)`: positive structural evidence, accumulated via exp-sum from neighbor pairs with matching relations whose confidence exceeds 0.5
- `neg_agg = 1 - exp(-λ × Σ neg_strengths)`: negative structural evidence, from neighbor pairs whose confidence is below 0.5 (weighted by forward functionality)
- The seed serves as a baseline: positive evidence pushes toward 1.0 proportional to the room above seed, negative evidence pushes toward 0.0 proportional to the seed value
- With no structural evidence, the fixpoint equals the seed (name-only matching preserved)

This replaces the dual-channel architecture (separate positive/negative channels combined via Bayesian log-odds) with a single score that directly integrates both types of evidence.

### Convergence in practice

For our sparse entity-relation graphs with functionality weights in [0, 1], the structural function's Lipschitz constant is typically well below 1 — each pair's score depends on a handful of neighbor pairs, each contributing through the saturating exp-sum. With damping `α = 0.5`, convergence is geometric with a practical rate around 0.3–0.5, reaching epsilon = 1e-4 within 10–15 iterations.

The `epsilon` convergence check and `max_iter` bound provide practical guarantees in all cases.

## Summary of evolution

| Aspect | Melnik 2002 | PARIS 2011 | FLORA 2025 | Ours |
|--------|-------------|------------|------------|------|
| **Framework** | Ad-hoc fixpoint | Probabilistic | Fuzzy logic | Damped fixpoint |
| **Evidence combination** | Sum + normalize | Noisy-OR (product of complements) | min within rule, max across rules | Exp-sum, seed-as-baseline |
| **Negative evidence** | None | Optional (Eq. 7, abandoned) | Excluded (breaks monotonicity) | Integrated (damping allows decrease) |
| **Relation weighting** | Inverse-product coefficients | [Functionality](functionality.md) (harmonic mean) | Functionality + local functionality | [Functionality](functionality.md) (global) |
| **Relation alignment** | Same labels only | Joint sub-relation discovery | Joint sub-relation discovery | Embedding similarity threshold |
| **Convergence** | Empirical (residual check) | Not proven | Proven (Knaster-Tarski) | Contraction mapping (Banach) |
| **Match selection** | Relative similarity + stable marriage | Maximum assignment per entity | Maximum assignment per entity | Union-find on threshold |

## References

- Melnik, Garcia-Molina, Rahm. *Similarity Flooding: A Versatile Graph Matching Algorithm and its Application to Schema Matching.* ICDE 2002.
- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* PVLDB 2011.
- Peng, Bonald, Suchanek. *FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.* 2025.
- Jeh, Widom. *SimRank: A Measure of Structural-Context Similarity.* KDD 2002.
- Lizorkin, Velikhov, Grinev, Turdakov. *Accuracy Estimate and Optimization Techniques for SimRank Computation.* PVLDB 2008. (Proposition 1: error bound `C^{k+1}`, formal proof of contraction convergence.)
- Brin, Page. *The Anatomy of a Large-Scale Hypertextual Web Search Engine.* WWW 1998. (PageRank damping as contraction.)
