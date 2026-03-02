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

A weakness of noisy-OR in practice is that a single high-confidence path can saturate the score, making one shared anchor indistinguishable from many. We address this with an [evidence factor](evidence_factor.md) that discounts structural scores supported by few paths.

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

## Summary of evolution

| Aspect | Melnik 2002 | PARIS 2011 | FLORA 2025 |
|--------|-------------|------------|------------|
| **Framework** | Ad-hoc fixpoint | Probabilistic | Fuzzy logic |
| **Evidence combination** | Sum + normalize | Noisy-OR (product of complements) | min within rule, max across rules |
| **Relation weighting** | Inverse-product propagation coefficients | [Functionality](functionality.md) (harmonic mean of local fun.) | Functionality + local functionality |
| **Relation alignment** | Same labels only | Joint sub-relation discovery | Joint sub-relation discovery |
| **Convergence** | Empirical (residual check) | Not proven | Proven (Knaster-Tarski) |
| **Dangling entities** | Not addressed | Handled implicitly | Explicit (non-matched entities stay at 0) |
| **Match selection** | Relative similarity + stable marriage | Maximum assignment per entity | Maximum assignment per entity |

## References

- Melnik, Garcia-Molina, Rahm. *Similarity Flooding: A Versatile Graph Matching Algorithm and its Application to Schema Matching.* ICDE 2002.
- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* PVLDB 2011.
- Peng, Bonald, Suchanek. *FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.* 2025.
