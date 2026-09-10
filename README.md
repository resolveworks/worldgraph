# Worldgraph

A proof-of-concept for knowledge extraction from news articles using cross-source structural matching.

## The Core Idea

News is redundant by nature — multiple outlets report the same events independently, using different wording but describing the same entities and facts. Worldgraph treats this redundancy as signal: facts reported across multiple independent sources are more likely to be true, and entities that appear in the same structural neighborhood across sources are likely the same entity.

## How It Works

### 1. Extract

Each article is processed independently by an LLM into a small graph of **entities** (people, organizations, places, things) and **events** (the facts the article asserts). An event is a reified fact — a node connected to its participants by roles:

```
Sarah Chen works at Nextera as CEO

        ┌──────────┐
        │ work at  │          event node — label is display text,
        └──────────┘          never used for matching
        │     │     │
     agent  patient capacity
        │     │     │
  Sarah Chen  Nextera  CEO
```

A second article covering the same fact — "Nextera employs Sarah Chen as chief executive" — produces the same structure with different labels. Extraction conventions pin one canonical shape per fact type (employment is person-anchored: `agent` is always the employee), so voice and perspective differences come out structurally identical.

### 2. Match

Entity pairs are **seeded from name similarity** (Soft TF-IDF + Jaro-Winkler); events carry no comparable names and start at a neutral prior. From there, **PARIS-style similarity propagation** decides everything structurally: a pair's confidence grows when its role-identical neighbors also match, weighted by per-role **functionality** — a role that nearly always maps a participant to a single event carries more evidence than a promiscuous one. Evidence from multiple paths aggregates as an exponential sum, rewarding breadth over any single strong path, while neighbors that fail to corroborate contribute negative evidence. The iteration is damped to a fixed point, and merges commit progressively through a single union-find gated on structural corroboration: names alone never merge anything, and pairs with no structurally tested neighbors never merge. Entities with no credible match are left dangling (following FLORA).

## Design Rationale

**Why reified events.** Making the fact itself a node lets events match through exactly the same mechanism as entities — participant structure. Paraphrase tolerance therefore lives where the evidence is (the participants), and the wording of the relation never has to be compared at all.

**Why a closed role vocabulary.** The matcher gates alignment on exact role equality — the assumption Similarity Flooding makes about edge labels. That only works if the vocabulary is closed and assigned consistently, so it is enforced in the schema, not just the prompt. The set is taken from the top of the intersection of the established inventories — PropBank, AMR, VerbNet, schema.org Action — where decades of independent annotation converge. Core roles (who bought, who was sold) are frame-specific and stay in the event label; only the universal periphery (places, capacities, beneficiaries, instruments) became roles. Consistency beats correctness: a philosophically wrong role assigned identically by every article matches perfectly; two defensible but different assignments never match.

## Open Problems

**Common structural templates.** Acquisitions, appointments, and earnings reports all produce similar subgraph shapes, so unrelated events can look alike topologically. The defense is that named entities from unrelated events don't match, so propagation between those graphs never fires — but this relies on names being sufficiently distinct.

**Commitment granularity.** "Acquired" and "announced plans to acquire" differ only in the event label, which is never matched. Two articles at different commitment levels with identical participants will structurally confirm each other.

**Source independence.** Wire services get rewritten in ways that look superficially independent. True source independence is hard to estimate.

## References

- Melnik, Garcia-Molina, Rahm. "Similarity Flooding: A Versatile Graph Matching Algorithm." ICDE 2002.
- Suchanek, Abiteboul, Senellart. "PARIS: Probabilistic Alignment of Relations, Instances, and Schema." PVLDB 2011.
- Liao, Sabetiansfahani, Bhatt, Ben-Hur. "IsoRankN: Spectral Methods for Global Alignment of Multiple Protein Networks." Bioinformatics 2009.
- Peng, Bonald, Suchanek. "FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic." ISWC 2025 (Best Paper).
- Chen et al. "What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings." ICML 2023.
