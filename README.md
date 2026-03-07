# Worldgraph

A proof-of-concept for knowledge extraction from news articles using cross-source structural matching.

## The Core Idea

News is redundant by nature — multiple outlets report the same events independently, using different wording but describing the same entities and relationships. Worldgraph treats this redundancy as signal: facts reported across multiple independent sources are more likely to be true, and entities that appear in the same relational neighborhood across sources are likely the same entity.

## How It Works

### 1. Extract

Each article is processed independently by an LLM to produce a small subgraph of entity-relation triples:

```
[Sarah Chen] —appointed as→ [CEO] —of→ [Nextera Inc]
[David Park] —resigned from→ [CEO] —of→ [Nextera Inc]
```

Another article covering the same event produces a structurally similar subgraph with different wording. The extraction doesn't need to be perfect — noise gets filtered in the next stage.

### 2. Match

Entity pairs across graphs are compared using **PARIS-style similarity propagation** (Suchanek et al., 2011), adapted for free-text relation phrases:

- Entity-entity confidence is **seeded from name similarity** (Soft TF-IDF + Jaro-Winkler) before propagation begins. This gives the iteration loop initial signal to work with — structurally connected neighbors that share similar names start with nonzero scores, which then propagate outward.
- Each iteration: propagate — a pair's score increases if their neighbors also score highly, weighted by relation phrase similarity and relation functionality (rare/specific relations carry more signal than generic ones). Neighbor pairs with zero confidence are skipped.
- Evidence from multiple paths is aggregated with an exponential sum: `1 - exp(-λ × Σ strengths)`. This naturally rewards breadth — a single strong path is heavily discounted (~0.63), while multiple paths accumulate proportionally.
- Repeat until scores converge, then threshold to decide which pairs to merge

Relations are compared via sentence embedding similarity — "acquired", "bought", "completed the purchase of" all cluster together without requiring a predefined schema. The standard Similarity Flooding algorithm (Melnik et al., 2002) requires identical edge labels to propagate similarity; we replace that binary gate with continuous relation-phrase similarity.

Entities with no credible match in another graph are simply left unmerged (dangling entities, following FLORA, Peng et al., 2025).

### 3. Score (planned)

Once matching is battle-tested, a scoring stage will rank each deduplicated fact by cross-source agreement — entities with multiple occurrences are matched entities, edges confirmed by multiple independent articles are higher-confidence facts. Provenance tracking through the pipeline makes this possible: every merged entity and edge retains its source articles.

## Algorithm Design

### The circularity problem

Entity resolution is inherently circular: to know if two entities are the same you need to know if they have the same relations to the same other entities — but resolving *those* entities has the same problem. Hard early decisions cascade: one wrong merge combines relationship sets and can trigger further wrong merges.

Similarity propagation dissolves this by never making hard decisions during propagation. Soft scores iterate to a fixpoint; a single threshold at the end produces the final merge decisions.

### Relation functionality

Not all relations carry equal evidence. "Is headquartered in" connects many companies to a few cities — knowing two entities share that relation weakly implies identity. "Signed a definitive merger agreement with" is rare and specific — two entities sharing that relation are almost certainly the same pair.

This is PARIS's *functionality* concept: a relation's weight is proportional to how often it maps a subject to a unique object. We approximate this with inverse average degree of the relation in the graph.

### Free-text relations

Standard methods (SF, PARIS, FLORA) assume relations come from a controlled vocabulary. We have free-text extraction, so every relation phrase is potentially unique. We handle this by pre-computing pairwise sentence-embedding similarity for all relation phrases and gating propagation paths on a cosine threshold — "acquired" and "purchased" pass (~0.85), "acquired" and "located in" don't. This replaces the exact-label-match gate in standard SF with a fuzzy one, but it's still a binary gate, not a continuous weight.

### N-graph alignment

Each article produces one graph. All article graphs are merged into a single unified graph, and propagation runs once over all cross-graph entity pairs simultaneously. Final matches are merged transitively via union-find.

## Open Problems

**Common structural templates.** Acquisitions, appointments, and earnings reports all produce similar subgraph shapes. Two unrelated acquisition events will have similar topology. The defense is that entity names from unrelated events won't match, so propagation between those graphs won't fire — but this relies on named entities being sufficiently distinct.

**Relation granularity.** "Acquired" vs. "announced plans to acquire" involves different levels of commitment. Embedding similarity doesn't distinguish these, and functionality weighting won't help either.

**Source independence.** Wire services get rewritten in ways that look superficially independent. True source independence is hard to estimate.

**Temporal dynamics.** Facts change over time. The system doesn't yet handle edges that should be timestamped or expired.

## References

- Melnik, Garcia-Molina, Rahm. "Similarity Flooding: A Versatile Graph Matching Algorithm." ICDE 2002.
- Suchanek, Abiteboul, Senellart. "PARIS: Probabilistic Alignment of Relations, Instances, and Schema." PVLDB 2011.
- Liao, Sabetiansfahani, Bhatt, Ben-Hur. "IsoRankN: Spectral Methods for Global Alignment of Multiple Protein Networks." Bioinformatics 2009.
- Peng, Bonald, Suchanek. "FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic." ISWC 2025 (Best Paper).
- Chen et al. "What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings." ICML 2023.
