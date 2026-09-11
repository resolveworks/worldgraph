# Worldgraph

Cross-source structural matching for knowledge extraction from news: outlets report the same facts in different wording; we extract term graphs per article — entities and recursively nested statements — and match their structure across sources, so entities deduplicate and facts confirm by multi-article agreement. Design narrative and literature: `README.md`.

## Commands

- Tests: `uv run --env-file .env pytest` — synthetic graphs, no LLM calls
- Lint / typecheck: `uv run ruff check .` and `uv run pyright`
- Pipeline: `uv run --env-file .env worldgraph extract articles/*.md -o graphs/`, then `uv run --env-file .env worldgraph match graphs/*.json -o matched.json` (match builds the predicate prior from `EMBEDDING_MODEL`)
- Eval (makes LLM calls): `uv run --env-file .env python evals/run.py`

## Why the design is what it is

Not derivable from the code — the reasoning behind it:

- **One recursive shape.** A term is an entity or a statement; a statement is a triple (subject term, predicate, object term), so statements about statements — qualifiers, attribution, denials — take the same shape with no special cases (RDF-star). Statements match through the same structural propagation as entities: "acquire" and "purchase" statements align because their participants align; predicates are compared only by a prior, which proposes and is never decisive.
- **Tolerance lives where the evidence is, not in extraction.** The previous schema (reified events + a closed role vocabulary) put paraphrase tolerance in extraction-side canonicalization, which demanded an annotation-guideline apparatus — golden extractions pinning a canonical shape per fact type — to keep consistent across articles. The term model absorbs predicate variation in the prior (a proposal the matcher structurally vetoes), participant variation in structure, and voice variation in a single rule: extraction normalizes to active voice, so the subject is the one who brings the fact about. The role taxonomy existed to replace embedding comparison; with priors back as proposals it is unnecessary. Never reintroduce per-fact-type canonicalization conventions — that way lies the guideline apparatus again.
- **Direction is the one irreducible decision, and it lives in the slots.** Antonym and inverse predicates are near-neighbors in embedding space, so orientation cannot live in similarity; it lives in subject/object position, where the evidence function sees it. The historical bug: adjacency that ignored which side of a triple a term sat on merged "X governs Y" with "Y governs X". Slot alignment makes the wrong-side search fail — negative evidence, never a merge. The known cost: inverse alternations ("works at" / "employs") stay crossed as parallel unconfirmed statements; the fix, if real data demands it, is a mechanical predicate-pair rule, not a taxonomy.
- **Similarity proposes, structure disposes.** Name seeds and predicate priors alone never merge anything; merging requires at least one structurally tested neighbor, and the union-find inside propagation is the sole merge authority. Positive evidence is shrunk by corroboration count — single-path pairs stay below the bar; negative evidence is full-weight — one mismatched participant is decisive.

## Testing

Tests are synthetic-graph spec tests, pinned in `tests/test_matcher.py`: each pins one property of the matcher (priors alone never merge, direction contradictions are negative evidence, paraphrase merges under a neutral prior, nesting propagates through the same mechanism). They encode the *expected* model, not current behavior — never tune an assertion to what the code currently returns. Red tests are an accepted state. One assumption per test, on the smallest input that shows it; no mocking. When a matching failure is identified, write the failing test before touching the algorithm.

Evals (`evals/`) are pair→merge and end-to-end: article pairs through the production extraction and matching, asserting what should merge. They make LLM calls and are expected to be red while the extraction prompt is untuned. Goldens are no longer annotation guidelines — the only extraction convention beyond the shape is the active-voice rule.

## Conventions

- **Clean refactors, not patches**: early-stage project, no external users. No back-compat shims, stale signatures, or dead code — rename and delete freely.
- **Pipeline modularity**: every stage runnable independently.
- **Scale-readiness**: decisions must hold up on real, noisy, multilingual, large-scale news feeds.
- **Docs state only what code can't show**: no regurgitating the codebase in docs or comments. When the algorithm, schema, or vocabulary changes, docs change in the same commit — stale docs are bugs.
