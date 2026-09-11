# Worldgraph

A proof-of-concept for knowledge extraction from news articles using cross-source structural matching.

## The Core Idea

News is redundant by nature — multiple outlets report the same events independently, using different wording but describing the same entities and facts. Worldgraph treats this redundancy as signal: facts reported across multiple independent sources are more likely to be true, and entities that appear in the same structural neighborhood across sources are likely the same entity.

## How It Works

### 1. Extract

Each article is processed independently by a single LLM pass into a graph of **terms**. A term is either an **entity** — a named thing in the world — or a **statement**: a triple of subject term, predicate, object term. Since statements are terms, a statement can be about a statement:

```
Sarah Chen works at Nextera as CEO

        ┌─────────────┐
        │ s1: work at │        a statement is a triple — and a term
        └─────────────┘        in its own right, so it can occupy
         │            │        another statement's slots
     subject        object
         │            │
    Sarah Chen    Nextera
         ▲
         │ subject
   ┌─────┴────┐           ┌─────┐
   │ s2:  as  │──object──▶│ CEO │
   └──────────┘           └─────┘
```

Qualifiers (a title, a place, a price, a scope) are statements about the statement they qualify; attribution and claims-about-claims are too, represented but never judged — a denial is `(denier, "denies", the denied statement)`, an allegation is `(claimant, "alleges", the claimed statement)`. The shape is RDF-star: nesting is ordinary structure, with no special cases.

The direction of a fact lives in subject/object position, and extraction normalizes to active voice so that the subject is the one who brings the fact about: "Acme acquired Beta", never the passive with the participants swapped. A second article — "Sarah Chen serves as chief executive of Nextera" — yields the same participants in the same slots with a different predicate; matching absorbs that. The prompt's other conventions keep the graph comparable across articles: entities are named things only, taken verbatim (unnamed mentions like "a cleaner" never become entities); pronouns resolve to their referents; the media itself — outlet, journalists, the act of reporting — is not part of the world and never appears.

### 2. Match

Entity pairs are **seeded from name similarity** (Soft TF-IDF + Jaro-Winkler, which survives abbreviations and acronyms); statement pairs are seeded from a **predicate-similarity prior** — smooth, pairwise, built from embeddings, 1.0 for identical predicates. The prior is a *proposal strength*, not a classification: it needs only to rank sensibly, because the matcher vetoes false proposals structurally.

From there, **damped pairwise propagation** (Similarity Flooding / PARIS family) decides everything structurally: a pair's confidence grows when its neighbors also match, and the core invariant is **slot alignment** — a statement's subject slot aligns only with the counterpart's subject slot, object with object. A participant in one slot finds counterparts only in that slot, so "X governs Y" versus "Y governs X" shows up as failed slot searches: negative evidence, never a merge. Nesting needs no special machinery — denials of the same claim lift each other through their object slots, qualifiers corroborate the statements they qualify through their subject slots, all by the same propagation.

Evidence is weighted by per-slot **functionality**: a participant that appears in the same slot of many statements is a weak identity signal, a rare one is strong. Multiple corroborations aggregate as an exponential sum, rewarding breadth over any single strong path, and positive evidence is shrunk by how many neighbors found counterparts — single-path pairs stay below the merge bar. Negative evidence carries full weight: one mismatched participant is decisive. The iteration runs to a fixed point and merges commit progressively through a single union-find gated on structural corroboration — names or priors alone never merge anything, and pairs with no structurally tested neighbors never merge. Terms with no credible counterpart are left unmerged (following FLORA).

### 3. Output

The pipeline emits the merged canonical graph: one term per match group — entities carry the union of their members' names as aliases, statements appear once with their endpoints remapped — plus the match groups themselves, as ids into the original article graphs. JSON I/O is strict: duplicate ids, unresolvable references, and unknown fields raise; invalid structure is never silently repaired.

## Design Rationale

**Why one recursive shape.** The previous schema reified facts as event nodes joined to participants by a closed role vocabulary. That made the fact matchable like an entity — the property worth keeping — but it put paraphrase tolerance in extraction-side canonicalization, which demanded an annotation-guideline apparatus (golden extractions pinning one canonical shape per fact type) to stay consistent across articles. The term model keeps facts as nodes with strictly less machinery — subject, predicate, object, recursion — and moves tolerance to where the evidence is: predicate variation is absorbed by the prior (a proposal the matcher can veto), participant variation by structure, voice variation by one active-voice rule. The closed role taxonomy existed to replace embedding comparison; with priors back as proposals, it is unnecessary.

**Why direction lives in the slots.** Extraction must make exactly one decision consistently: which participant is the subject. Antonym and inverse predicates are near neighbors in embedding space, so orientation cannot live in similarity — it lives in subject/object position, where the evidence function sees it. The historical bug was adjacency that ignored which side of a triple a term sat on: it merged "X governs Y" with "Y governs X". Slot alignment turns that coincidence into failed counterpart searches — negative evidence.

## Open Problems

**Inverse-predicate alternations.** "Jane works at Supercorp" and "Supercorp employs Jane" are both active voice, describe the same fact, and place the participants in opposite slots. Slot alignment correctly refuses to merge them, but the result is two parallel unconfirmed statements rather than one confirmed fact. If real data demands it, the fix is a mechanical predicate-pair rule — a small table of inverse pairs — not a taxonomy.

**Common structural templates.** Acquisitions, appointments, and earnings reports all produce similar subgraph shapes, so unrelated statements can look alike topologically. The defense is that named entities from unrelated facts don't match, so propagation between those graphs never fires — but this relies on names being sufficiently distinct.

**Commitment granularity.** "Acquired" and "announced plans to acquire" differ only in the predicate, and identical participants can merge on structure alone. Two articles at different commitment levels will structurally confirm each other.

**Source independence.** Wire services get rewritten in ways that look superficially independent. True source independence is hard to estimate.

## References

- Melnik, Garcia-Molina, Rahm. "Similarity Flooding: A Versatile Graph Matching Algorithm." ICDE 2002.
- Suchanek, Abiteboul, Senellart. "PARIS: Probabilistic Alignment of Relations, Instances, and Schema." PVLDB 2011.
- Liao, Sabetiansfahni, Bhatt, Ben-Hur. "IsoRankN: Spectral Methods for Global Alignment of Multiple Protein Networks." Bioinformatics 2009.
- Peng, Bonald, Suchanek. "FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic." ISWC 2025 (Best Paper).
- Chen et al. "What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings." ICML 2023.
- W3C. "RDF 1.2 Concepts and Abstract Data Model" — triple terms, the statements-about-statements shape (formerly RDF-star).
