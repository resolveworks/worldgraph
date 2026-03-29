# Worldgraph

## What This Is

A proof-of-concept for **cross-source structural matching** for knowledge extraction from news articles. See `README.md` for the full proposal.

The core idea: news redundancy is the signal, not noise. Multiple outlets report the same facts with different wording. By extracting small entity-relation subgraphs per article and matching their structure across sources, we can deduplicate entities and validate facts more reliably than string matching alone.

The target input is a continuous feed of all major news outlets — not a curated sample. The PoC uses synthetic test data for controlled evaluation, but algorithmic decisions should hold up at that scale. A PoC that only works on hand-picked articles proves nothing.

## Architecture (PoC Pipeline)

1. **Extract** — Process each article independently with an LLM to produce entity-relation subgraphs (one JSON per article)
2. **Match** — Align entities across graphs using similarity propagation

```bash
worldgraph extract articles/*.json -o graphs/    # article JSON → per-article graph JSON
worldgraph match graphs/*.json -o matched.json   # per-article graphs → unified matched graph
```

In the matched output, entities with >1 occurrence are matched entities, edges with >1 article are confirmed facts.

## Project Structure

- `worldgraph/` — one module per pipeline stage (`extract.py`, `match.py`), plus `cli.py` for the Click entry point
- `docs/` — detailed algorithm write-ups referenced from this file
- `tests/` — pytest suite, layered (see Testing Strategy below)

## Matching Algorithm

The matching stage implements **similarity propagation** (inspired by PARIS/FLORA) adapted for free-text relation phrases. The design is driven by the literature and validated by automated tests — we don't have real-world ground truth yet, so the tests encode what the algorithm *should* do based on the papers and our understanding of the domain. When a test fails, it's either a bug or a wrong assumption about what the algorithm needs.

See `docs/` for detailed write-ups on each concept referenced below.

### What's implemented

The core propagation loop (`match.py`):

1. **Name-similarity seeding** — Soft TF-IDF + Jaro-Winkler seeds the confidence dict before iteration starts. This gives propagation initial signal to work with. See [docs/name_similarity.md](docs/name_similarity.md).

2. **Relation similarity via sentence embeddings** — relation phrase similarity is a continuous multiplier on propagation paths, not a binary gate. "acquired" ↔ "purchased" (~0.85) contributes proportionally; "acquired" ↔ "located in" (~0.1) contributes almost nothing. This replaces the identical-label requirement in standard SF/PARIS.

3. **Functionality weighting** — global forward and inverse functionality (1/avg_degree), with similar relation phrases pooled. See [docs/functionality.md](docs/functionality.md).

4. **Exponential sum aggregation** — `1 - exp(-λ × Σ strengths)` where each path contributes `rel_sim × min(func_a, func_b) × neighbor_confidence`. Rewards breadth over single strong paths.

5. **Monotone non-decreasing updates** — confidence only goes up, never down. Preserves convergence guarantees (FLORA-style).

6. **Unified N-graph matching** — all article graphs merged into one, propagation runs once over all cross-graph pairs. Final grouping via union-find.

7. **Negative evidence** ([docs/negative_evidence.md](docs/negative_evidence.md)) — dampened negative factor penalizes entity pairs whose functional neighbors don't match. Applied once per convergence cycle, after positive evidence stabilizes. Uses name-seed confidence (not structural) for neighbor matching to prevent circular reinforcement.

8. **Progressive merging** ([docs/progressive_merging.md](docs/progressive_merging.md)) — high-confidence merges are committed inline during the single propagation loop. Canonical adjacency is updated incrementally on merge (O(degree) per merge), avoiding full adjacency rebuilds. Enriched neighborhoods compound structural evidence across merge cycles.

### What's not implemented (yet)

These are documented in `docs/` with design sketches but no code.

- **Local functionality** — FLORA uses per-entity functionality (`1/|targets for this specific source|`), not just global averages. We only compute global.

- **Confidence-weighted union-find** ([docs/multi_graph_alignment.md](docs/multi_graph_alignment.md)) — current union-find enforces blind transitivity. A↔B and B↔C above threshold merges all three regardless of A↔C score. Validating group coherence would catch the worst cascading false merges.

- **Cross-lingual support** ([docs/cross_lingual.md](docs/cross_lingual.md)) — swapping to a multilingual embedding model. The algorithm is language-agnostic by design; only the model choice needs to change.

- **Joint relation alignment** — PARIS alternates entity and relation alignment. We pre-compute relation similarity from embeddings and hold it fixed.

## Tech Stack

- Python 3.12, managed with **uv** (`uv run`, `uv add`, etc.)
- LLM (Claude API) for entity/relation extraction
- Sentence embeddings for relation phrase similarity; Soft TF-IDF + Jaro-Winkler for entity name matching
- No ML training — fully unsupervised, classical graph methods

## Testing Strategy

One assumption per test, on the smallest input where it's observable. No mocking — real embeddings throughout. A failure at a higher layer should always be explainable by a failure at a lower layer.

Session-scoped fixture in `conftest.py`: `embedder` provides a session-scoped `Embedder` instance. Use `embedder.embed(keys, template=RELATION_TEMPLATE)` for relation embeddings.

- **Layer 1 — Unit**: individual primitives (`compute_functionality`, `UnionFind`)
- **Layer 2 — Propagation**: structural and functionality effects on similarity scores, convergence guarantees
- **Layer 3 — Integration**: correct merges, no spurious matches, correct canonical names and confirmed edges

## Conventions

- **Clean refactors, not patches**: this is an early-stage project with no external users. Every change should produce a pristine new state — never add backward-compatibility shims, preserve stale signatures, or keep dead code around "just in case". Refactor completely: rename freely, change interfaces, delete old code. No technical debt.
- **CI/GitHub Actions**: when working on an issue (not already on a PR), always create a pull request with `gh pr create` after pushing your branch. Never just post a compare link — open the actual PR.
- **Pipeline modularity**: each stage should be runnable independently.
- **Scale-readiness**: the core algorithm must hold up on real, noisy, multilingual, large-scale news data.
- **Tests before fixes**: when a matching failure is identified, write a failing test (xfail if needed) that captures the specific scenario before changing the algorithm. The test encodes what "correct" means; the fix is just making it pass.
