# Worldgraph

## What This Is

A proof-of-concept for **cross-source structural matching** for knowledge extraction from news articles. See `README.md` for the full proposal.

The core idea: news redundancy is the signal, not noise. Multiple outlets report the same facts with different wording. By extracting small entity-relation subgraphs per article and matching their structure across sources, we can deduplicate entities and validate facts more reliably than string matching alone.

The target input is a continuous feed of all major news outlets — not a curated sample. The PoC uses synthetic test data for controlled evaluation, but algorithmic decisions should hold up at that scale. A PoC that only works on hand-picked articles proves nothing.

## Architecture (PoC Pipeline)

1. **Extract** — Process each article independently with an LLM to produce entity-relation subgraphs → `data/graphs/{article_id}.json`
2. **Match** — Align entities across graphs using similarity propagation → `data/matched.json`
3. **Score** — Score each deduplicated fact by cross-source agreement

```bash
worldgraph extract                               # data/articles/ → data/graphs/
worldgraph match                                 # data/graphs/  → data/matched.json
```

In `matched.json`, entities with >1 occurrence are matched entities, edges with >1 article are confirmed facts.

## Project Structure

```
data/
  articles/               # Input: one {uuid}.json per article
  graphs/                 # Output of stage 1: one {article_id}.json per article
  matched.json            # Output of stage 2: merged graphs
worldgraph/
  __init__.py
  cli.py                  # Click CLI entry point (worldgraph command group)
  extract.py              # Stage 1: LLM-based entity/relation extraction → graph JSON
  match.py                # Stage 2: Structural matching + graph merging
.env.example              # Template for API key configuration
pyproject.toml            # Project config, dependencies, CLI entry point
README.md                 # Project proposal / algorithm design
CLAUDE.md                 # This file
```

## Test Data

`data/articles/` contains synthetic news articles (one `{uuid}.json` per article) covering multiple independent event clusters. Each cluster has several outlets reporting the same underlying facts with different wording.

The articles are designed to exercise the algorithm's key challenges:

- **Entity name variation**: same entity referred to differently across articles (e.g. "Meridian Technologies" / "Meridian Tech", "Dr. Priya Sharma" / "P. Sharma")
- **Relation phrasing variation**: same fact expressed with different verbs/phrases (e.g. "acquired" / "buys" / "purchase")
- **Cross-event entity linking**: entities that appear in multiple event clusters
- **Dangling entities**: most entities in any given article have no counterpart in most other articles

## Matching Algorithm

The matching stage implements **similarity propagation** (inspired by PARIS/FLORA) adapted for free-text relation phrases.

### Key concepts from the literature

**Similarity Flooding** (Melnik et al., 2002): entity similarity propagates through graph structure iteratively. To know if two entities match you need to know if their neighbors match — propagation dissolves this circularity by never making hard early decisions.

**PARIS** (Suchanek et al., 2011): extends SF to knowledge base alignment with *functionality weighting* — a relation's contribution to entity similarity is scaled by how functional it is (how often it maps a subject to a unique object). Rare/specific relations carry more signal than generic ones.

**FLORA** (Peng, Bonald, Suchanek, 2025): PARIS successor using fuzzy logic (t-norms/t-conorms) instead of probability, with proven convergence and explicit dangling-entity handling.

### Our adaptations

Standard SF/PARIS assume a shared or alignable relation vocabulary. We have free-text phrases. Adaptations:

1. **Relation similarity via sentence embeddings**: instead of requiring identical edge labels to allow propagation, gate propagation paths by cosine similarity of relation phrase embeddings. Only paths where rel_sim >= threshold propagate; "acquired" and "buys" pass (~0.85), "acquired" and "located in" don't.

2. **Functionality from phrase frequency**: a relation phrase appearing as the unique connection between two specific entities is maximally specific. Approximate PARIS functionality as inverse average degree of the relation in the graph.

3. **Dangling entities by default**: most entities won't match anything across most graph pairs. Threshold-based finalization naturally leaves them unmerged — no special handling needed.

### Algorithm sketch

For each pair of graphs (Gi, Gj):
1. **Name similarity** (fixed): `name_sim[(ei, ej)] = dot(name_emb(ei), name_emb(ej))` for all entity pairs. Computed once, never updated.
2. **Structural propagation**: structural scores start at 0. Each iteration, for each entity pair (ei, ej), examine all edge pairs (ei→ei' via r, ej→ej' via r') where `rel_sim(r, r') >= threshold` and the neighbor pair's confidence `name_sim + structural >= threshold`. Update via max: `structural[(ei, ej)] = max(neighbor_confidence * rel_sim * functionality)`. Scores are monotonically non-decreasing — convergence guaranteed (FLORA / Knaster-Tarski fixpoint).
3. **Select matches**: keep pairs where both `name_sim >= threshold` and `structural >= threshold`.
4. **Merge** matched pairs transitively via union-find.

Optionally: run single propagation over the full K-partite entity-pair graph (all N graphs simultaneously) to get transitive matches without O(N²) pairwise passes.

### What we don't do (yet)

- PARIS-style joint relation alignment loop (relation similarities updated from entity similarities, alternately) — we pre-compute relation similarity from embeddings and hold it fixed
- The full IsoRankN spectral clustering for N-graph alignment — we run pairwise and merge transitively

## Tech Stack

- Python 3.12, managed with **uv** (`uv run`, `uv add`, etc.)
- LLM (Claude API) for entity/relation extraction
- Sentence embeddings (fastembed) for name and relation phrase similarity
- No ML training — fully unsupervised, classical graph methods

## Testing Strategy

One assumption per test, on the smallest input where it's observable. No mocking — real embeddings throughout. A failure at a higher layer should always be explainable by a failure at a lower layer.

A session-scoped `embed_phrase(phrase)` fixture (in `conftest.py`) embeds relation phrases on demand and caches results — each unique phrase is embedded at most once per session.

- **Layer 1 — Unit**: individual primitives (`cosine_sim`, `compute_functionality`, `select_matches`, `UnionFind`)
- **Layer 2 — Propagation**: structural and functionality effects on similarity scores, convergence guarantees
- **Layer 3 — Integration**: correct merges, no spurious matches, correct canonical names and confirmed edges

## Conventions

- Keep the pipeline modular — each stage should be runnable independently
- Implementation shortcuts are fine (no error handling, no production polish) — but the core algorithm must be designed to hold up on real, noisy, large-scale news data
- All intermediate outputs should be inspectable (write to JSON files between stages)
