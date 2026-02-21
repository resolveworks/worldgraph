# Cross-Source Structural Matching for Knowledge Extraction

_A proposal for discussion — February 2026_

## The Core Idea

News is redundant by nature. When a major event happens, dozens of outlets report the same facts independently, using different wording but describing the same entities and relationships. Current knowledge extraction systems treat this redundancy as a problem to be filtered out. We think it's the primary signal.

We propose a system built on two reinforcing principles:

- **Cross-source validation:** Facts reported by multiple independent sources are likely true. Extraction errors are random and don't repeat across sources. A fact appearing in five different articles is far more reliable than one appearing in a single source.

- **Structural deduplication:** Rather than resolving entities by name matching alone, we use the shape of relationships around them. When two entities have the same relational neighborhood across multiple documents, they're almost certainly the same entity, even if their surface names differ.

These two principles create an iterative loop: resolving entities reveals more shared relationships, which validates more facts, which reveals more entity overlaps. The system bootstraps itself into progressively cleaner knowledge.

## How It Works

### 1. Extract Small Subgraphs Per Article

Each article is processed independently to produce a small subgraph of entity-relation triples. For example, an article about a CEO change might produce:

> `[Sarah Chen] —appointed→ [CEO] —at→ [Nextera Inc]` > `[David Park] —resigned→ [CEO] —at→ [Nextera Inc]`

Another article covering the same event produces a similar subgraph with different wording: "named as chief executive," "stepping down from." The extraction layer doesn't need to be perfect. Noise gets filtered later.

### 2. Overlay and Match Subgraph Structure

This is the key step. Rather than comparing individual entity names across articles, we look for subgraphs that have the same shape. When 15 articles all produce a subgraph with the topology `[Person] → [Role] → [Organization]` involving fuzzy-matching names and overlapping time windows, that structural convergence is a much stronger deduplication signal than any string comparison on individual entities.

The intuition: a single name match is ambiguous ("Jordan" could be many people). But when "Jordan" appears in the same structural neighborhood—connected to the same organization, role, and event—across multiple independent sources, the ambiguity collapses.

### 3. Merge Relations via Embedding Similarity

Different articles express the same relationship in different ways: "acquired," "bought," "completed the purchase of." We handle this by embedding relation phrases into a vector space (using a sentence embedding model) and clustering similar relations together. This gives us a normalized relation vocabulary without requiring a predefined schema. The clustering threshold can start conservative and loosen as structural overlap provides additional confidence.

### 4. Iterate Until Convergence

Entity resolution and relation clustering reinforce each other in a loop:

- Merge high-confidence entity pairs (strong string similarity + structural overlap).
- Each merge combines relationship sets, potentially revealing new overlaps.
- New overlaps surface additional entity matches that weren't visible before.
- Repeat until no new merges meet the confidence threshold.

The process starts with safe, obvious merges and progressively resolves harder cases as the graph becomes cleaner.

### 5. Score Facts by Cross-Source Agreement

After deduplication, each unique fact is scored by how many independent sources report it. Source independence can be estimated from the data itself: outlets that consistently publish identical facts with similar timing are likely dependent (wire services, syndication). True validation comes from genuinely independent reporting.

## Why Structural Matching

This approach has precedent outside of news. In computational biology, protein interaction networks are aligned across species by matching interaction topology—if a cluster of proteins has the same structural pattern in humans and mice, they're likely orthologs. In social network de-anonymization, researchers showed that the topology of a person's friend graph is essentially a fingerprint, sufficient to identify them across platforms without any name information at all.

The key insight these domains share with news: entities are defined by their relationships. Two nodes with the same relational neighborhood are the same entity, regardless of what they're labeled. News subgraphs are smaller and shallower than biological or social networks, but the compensating factor is volume—thousands of articles producing overlapping subgraphs daily.

This also flips the conventional pipeline. Standard approaches resolve entities first (by name), then build a graph. We propose building many small provisional graphs first, then using their structural overlap to resolve entities. The graph becomes the deduplication mechanism rather than something constructed after deduplication is already done.

## Open Problems

These are the hard problems we see. We'd like to discuss whether they're tractable.

**Common structural templates.** News is full of repeated event patterns: acquisitions, appointments, earnings reports. These produce nearly identical subgraph shapes for entirely different events. Structural matching alone will generate false positives between unrelated events that happen to share the same topology. Temporal windowing helps—but how much?

**Error cascading.** The iterative merge loop is powerful but potentially fragile. A bad early merge could propagate: incorrectly merging two entities combines their relationships, which could trigger further incorrect merges. What are the convergence properties? Can we bound error propagation? Should merges be reversible?

**Sparse entities.** A large portion of entities in news appear in only one or two articles—exactly where structural signal is weakest. These entities can't be validated through cross-source overlap. Do they simply remain unvalidated, or is there a fallback strategy that doesn't collapse back to pure string matching?

**Relation granularity.** "Acquired" vs. "bought" is straightforward for embedding similarity. But "is expanding into" vs. "announced plans to enter the market of" involves different levels of commitment. How fine-grained should relation equivalence be? Does the clustering approach handle these edge cases, or do we need explicit relation hierarchies?

**Source independence.** Detecting whether sources are truly independent is hard in practice. Wire services get rewritten in ways that look superficially independent. Estimating dependency from co-occurrence patterns is promising but unproven at scale.

**Temporal dynamics.** Facts change. CEOs turn over, companies merge, relationships evolve. The system needs to handle the same entity having different relationships at different times without conflating them. How should edges be timestamped and expired?

## Next Steps

We plan to build a proof of concept: a Python pipeline that takes a set of news articles about overlapping events, extracts subgraphs (using an LLM for extraction), clusters relations via sentence embeddings, runs the iterative structural matching loop, and outputs a deduplicated, confidence-scored knowledge graph.

The goal is to demonstrate the core loop on real data: does structural overlap actually resolve entities that string matching misses? Does cross-source agreement filter extraction noise effectively? Where does it break?

We'd welcome input on whether this is a promising direction, which open problems are most critical, and what existing work we should be building on.

---

_For discussion. Feedback welcome._
