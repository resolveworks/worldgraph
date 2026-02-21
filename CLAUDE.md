# Worldgraph

## What This Is

A proof-of-concept for **cross-source structural matching** for knowledge extraction from news articles. See `README.md` for the full proposal.

The core idea: news redundancy is the signal, not noise. Multiple outlets report the same facts with different wording. By extracting small entity-relation subgraphs per article and matching their structure across sources, we can deduplicate entities and validate facts more reliably than string matching alone.

## Architecture (PoC Pipeline)

The pipeline has 5 stages:

1. **Extract** — Process each article independently with an LLM to produce entity-relation subgraphs (triples)
2. **Embed & Cluster Relations** — Embed relation phrases with a sentence embedding model, cluster similar relations ("acquired" ≈ "bought" ≈ "completed the purchase of")
3. **Structural Matching** — Overlay per-article subgraphs; find isomorphic structures with fuzzy-matching node labels across sources
4. **Iterative Merge** — Loop: merge high-confidence entity pairs → combine relationship sets → discover new overlaps → repeat until convergence
5. **Score** — Score each deduplicated fact by cross-source agreement (number of independent sources reporting it)

## Project Structure

```
data/articles.json    # Input: fake news articles for testing (10 articles, 3 overlapping events)
README.md             # Project proposal / discussion document
CLAUDE.md             # This file
```

## Test Data

`data/articles.json` contains 10 synthetic news articles covering 3 events:

- **Acquisition**: Meridian Technologies acquires Lightwave Analytics ($2.3B)
- **CEO change**: Sarah Chen replaces David Park at Nextera Energy Solutions
- **Partnership**: Meridian Technologies + Cascade Robotics AI automation venture ($500M)

The articles have deliberate variation in entity naming (e.g. "Meridian Technologies" / "Meridian Tech" / "the Palo Alto company", "James Xu" / "J. Xu", "Dr. Priya Sharma" / "P. Sharma") and relation phrasing ("acquired" / "buys" / "purchase"). Articles 8 and 9 cross-reference multiple events, testing cross-event entity linking.

## Tech Stack

- Python 3.12, managed with **uv** (`uv run`, `uv add`, etc.)
- LLM (Claude API) for entity/relation extraction
- Sentence embeddings for relation clustering
- Stack otherwise TBD — keep it simple, this is a proof of concept

## Conventions

- Keep the pipeline modular — each stage should be runnable independently
- Prefer simplicity over robustness; this is an exploration, not production code
- All intermediate outputs should be inspectable (write to JSON files between stages)
