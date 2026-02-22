# Worldgraph

## What This Is

A proof-of-concept for **cross-source structural matching** for knowledge extraction from news articles. See `README.md` for the full proposal.

The core idea: news redundancy is the signal, not noise. Multiple outlets report the same facts with different wording. By extracting small entity-relation subgraphs per article and matching their structure across sources, we can deduplicate entities and validate facts more reliably than string matching alone.

## Architecture (PoC Pipeline)

The pipeline uses a unified **graph format** (`{"graphs": [...]}`) that flows between stages. The extract stage outputs this format directly, and the match stage consumes and produces it, so iteration is just calling match repeatedly.

1. **Extract** — Process each article independently with an LLM to produce entity-relation subgraphs, output directly as graph JSON → `graphs_0.json`
2. **Structural Matching** — Match entity/edge structures across graphs, merge overlapping graphs → `graphs_N.json` (run repeatedly until convergence). Entities are compared by name embedding similarity; relations are compared by phrase embedding similarity (original phrases preserved, no flattening).
3. **Score** — Score each deduplicated fact by cross-source agreement (number of independent sources reporting it)

```bash
worldgraph extract                               # data/articles/ → graphs_0.json
worldgraph match                                 # graphs_0.json → graphs_1.json
worldgraph match -i data/graphs_1.json -o data/graphs_2.json  # repeat until stable
```

In the graph format, entities with >1 occurrence are matched entities, edges with >1 article are confirmed facts.

## Project Structure

```
data/
  articles/               # Input: one {uuid}.json per article
  graphs_0.json           # Output of stage 1 (per-article graphs with original relation phrases)
  graphs_N.json           # Output of stage 2 iterations (merged graphs, N = iteration number)
worldgraph/
  __init__.py
  cli.py                  # Click CLI entry point (worldgraph command group)
  extract.py              # Stage 1: LLM-based entity/relation extraction → graph JSON
  match.py                # Stage 2: Structural matching + graph merging (iterative)
.env.example              # Template for API key configuration
pyproject.toml            # Project config, dependencies, CLI entry point
README.md                 # Project proposal / discussion document
CLAUDE.md                 # This file
```

## Test Data

`data/articles/` contains synthetic news articles (one `{uuid}.json` per article) covering multiple independent event clusters. Each cluster has several outlets reporting the same underlying facts with different wording.

The articles are designed to exercise the algorithm's key challenges:

- **Entity name variation**: same entity referred to differently across articles (e.g. "Meridian Technologies" / "Meridian Tech", "Dr. Priya Sharma" / "P. Sharma")
- **Relation phrasing variation**: same fact expressed with different verbs/phrases (e.g. "acquired" / "buys" / "purchase")
- **Cross-event entity linking**: entities that appear in multiple event clusters, requiring resolution across clusters
- **Iterative merging**: transitive chains that only fully resolve after earlier merges establish intermediate links

## Tech Stack

- Python 3.12, managed with **uv** (`uv run`, `uv add`, etc.)
- LLM (Claude API) for entity/relation extraction
- Sentence embeddings for entity name and relation phrase comparison during matching
- Stack otherwise TBD — keep it simple, this is a proof of concept

## Conventions

- Keep the pipeline modular — each stage should be runnable independently
- Prefer simplicity over robustness; this is an exploration, not production code
- All intermediate outputs should be inspectable (write to JSON files between stages)
