# Worldgraph

Cross-source structural matching for knowledge extraction from news: outlets report the same facts in different wording; we extract entity–event graphs per article and match their structure across sources, so entities deduplicate and facts confirm by multi-article agreement. Design narrative and literature: `README.md`.

## Commands

- Tests: `uv run pytest` — requires `EXTRACTION_MODEL` in `.env` (the suite imports `evals/run.py`, which reads it at import time; no LLM calls are made)
- Lint / typecheck: `uv run ruff check .` and `uv run pyright`
- Pipeline: `uv run worldgraph extract articles/*.md -o graphs/`, then `uv run worldgraph match graphs/*.json -o matched.json`
- Extraction eval (makes LLM calls): `uv run python evals/run.py`

## Why the design is what it is

Not derivable from the code — the reasoning behind it:

- **Relations are reified as event nodes** so that events match through the same structural propagation as entities: "acquired" and "purchased" events align because their participants align — event labels are display text and are never compared.
- **The closed role vocabulary satisfies Similarity Flooding's identical-edge-label assumption by construction**, which is what lets the matcher gate alignment on exact role equality. The set is not arbitrary: it is the top of the intersection of the established inventories (PropBank's arguments/ArgM set, AMR's relations, VerbNet's thematic roles, schema.org Action), trimmed to what news text exercises. Core roles are frame-specific and live in the event label; only the periphery — where those projects converge — became roles.
- **Consistency beats correctness** in role assignment: what breaks matching is not a philosophically wrong role but two articles assigning *different* roles to the same participant. Every added role is another LLM decision point; the set grows only when a failing test or golden demands a distinction it cannot express. Never extend it casually.
- **Extraction canonicalization**, pinned by tests and goldens: stative employment is person-anchored (`agent`=employee, `patient`=employer, `capacity`=title) regardless of the article's voice, so "X works at Acme" and "Acme employs X" yield identical structure. Hiring is a different fact — an employer's action — and must never merge with a stative works-at event. Organizations are patients, not locations; facilities and cities are locations/destinations.

## Testing

Tests and eval goldens encode the *expected* model, not current behavior — never tune an assertion to what the code currently returns. Red tests are an accepted state. Goldens double as the annotation guidelines for role assignment. One assumption per test, on the smallest input that shows it; no mocking. When a matching failure is identified, write the failing test before touching the algorithm.

## Conventions

- **Clean refactors, not patches**: early-stage project, no external users. No back-compat shims, stale signatures, or dead code — rename and delete freely.
- **Pipeline modularity**: every stage runnable independently.
- **Scale-readiness**: decisions must hold up on real, noisy, multilingual, large-scale news feeds.
- **Docs state only what code can't show**: no regurgitating the codebase in docs or comments. When the algorithm, schema, or vocabulary changes, docs change in the same commit — stale docs are bugs.
