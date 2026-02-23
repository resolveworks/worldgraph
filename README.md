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

Entity pairs across graphs are compared using similarity propagation (following Melnik et al., "Similarity Flooding", 2002):

- Initialize similarity scores from name embeddings
- Each iteration: boost a pair's score if their neighbors also score highly, weighted by relation specificity (rare relations carry more signal than common ones)
- Repeat until scores converge, then threshold to decide which pairs to merge

Relations are matched by embedding similarity — "acquired", "bought", "completed the purchase of" all cluster together without requiring a predefined schema.

### 3. Score

After merging, each unique fact is scored by how many independent sources report it. Entities with multiple occurrences are matched entities; edges with multiple source articles are confirmed facts.

## Open Problems

**Common structural templates.** Acquisitions, appointments, and earnings reports all produce similar subgraph shapes. Structural matching alone risks false positives between unrelated events that share the same topology.

**Sparse entities.** Entities appearing in only one or two articles have weak structural signal and can't be validated through cross-source overlap.

**Relation granularity.** "Acquired" vs. "announced plans to acquire" involves different levels of commitment. Embedding similarity doesn't distinguish these.

**Source independence.** Wire services get rewritten in ways that look superficially independent. True source independence is hard to estimate.

**Temporal dynamics.** Facts change over time. The system doesn't yet handle edges that should be timestamped or expired.
