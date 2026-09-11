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
- **Tolerance lives where the evidence is, not in extraction.** Paraphrase tolerance must live somewhere: outlets word the same fact differently, so wording cannot be trusted as evidence. Housing it in extraction — canonical shapes or vocabularies the extractor must apply identically in every article — makes consistency the bottleneck, because each convention is another LLM decision point where two articles diverge structurally, and holding the line takes an annotation-guideline apparatus. The term model keeps extraction to a single convention (active voice) and puts tolerance where the evidence is: predicate variation is absorbed by the prior (a proposal the matcher structurally vetoes), participant variation by structure. Never add per-fact-type canonicalization conventions to the prompt — that way lies the guideline apparatus.
- **Direction is the one irreducible decision, and it lives in the slots.** Antonym and inverse predicates are near-neighbors in embedding space, so orientation cannot be recovered from a similarity score; it lives in subject/object position, where the evidence function gates on it. Participant adjacency without slot awareness would merge "X governs Y" with "Y governs X"; slot alignment makes the wrong-side search fail — negative evidence, never a merge. The accepted cost: inverse alternations ("works at" / "employs") stay crossed as parallel unconfirmed statements; the remedy, if real data demands one, is a mechanical predicate-pair rule — a small table of inverse pairs — not a vocabulary.
- **Similarity proposes, structure disposes.** Name seeds and predicate priors alone never merge anything; merging requires at least one structurally tested neighbor, and the union-find inside propagation is the sole merge authority. Positive evidence is shrunk by corroboration count — single-path pairs stay below the bar; negative evidence is full-weight — one mismatched participant is decisive.
- **Merged statements keep every attested predicate wording.** Mirroring entity name aliases: merging unions the members' predicates rather than keeping the representative's, so wording disagreement between sources — "acquire" vs "purchase" — stays visible in the merged statement's label set instead of being silently adjudicated at merge.

## Testing

Tests are synthetic-graph spec tests: each pins one property of the matcher (priors alone never merge, direction contradictions are negative evidence, paraphrase merges under a neutral prior, nesting propagates through the same mechanism). They encode the *expected* model, not current behavior — never tune an assertion to what the code currently returns. Red tests are an accepted state. One assumption per test, on the smallest input that shows it; no mocking. The suite is a ladder, built from principles upward: unit mechanics, single-assumption minimal graphs, composite scenarios where several mechanisms interact, and full multi-article situations asserting the complete canonical outcome. When a matching failure is identified, write the failing test before touching the algorithm.

Evals (`evals/`) evaluate extraction only — the LLM stage; the matcher is deterministic and covered by tests. Each case is one article with a golden of entities plus nested facts in surface form (subject name or nested fact, base-form predicate, object); comparison is exact on names and predicate strings, never on ids, reporting precision/recall with misses and extras. The suite is a ladder: one-sentence base triples (an active/passive pair pinning the active-voice convention), single constructions (a qualifier, a denial), a short multi-fact article, a paraphrase pair judged as independent goldens. Evals make LLM calls and are expected to be red while the extraction prompt is untuned — the diffs are the prompt-tuning agenda. The extraction conventions are active voice and base-form predicates; there are no annotation guidelines beyond that.

## Conventions

- **Clean refactors, not patches**: early-stage project, no external users. No back-compat shims, stale signatures, or dead code — rename and delete freely.
- **Pipeline modularity**: every stage runnable independently.
- **Scale-readiness**: decisions must hold up on real, noisy, multilingual, large-scale news feeds.
- **Docs state only what code can't show**: no regurgitating the codebase in docs or comments. When the algorithm, schema, or vocabulary changes, docs change in the same commit — stale docs are bugs.
