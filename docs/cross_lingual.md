# Cross-Lingual Matching

The target is a continuous feed of all major news outlets worldwide. Articles in many languages report the same events. This document explains how the matching algorithm handles language differences without language-specific mechanisms.

## Design principle: structure is the universal signal

The matching algorithm has two sources of evidence for entity equivalence:

1. **Name similarity** — Soft TF-IDF + Jaro-Winkler on entity labels
2. **Structural similarity** — propagated through the graph via shared neighbors and relations

Name similarity is language-dependent. It works within a script (English ↔ English, Chinese ↔ Chinese) but fails across scripts ("Tokyo" ↔ "東京" share zero characters). Rather than patching this with transliteration or other language-specific hacks, we treat cross-script matching as the case where name evidence is simply absent — and structural evidence must be sufficient on its own.

This is not a compromise. Structural matching is the core algorithm; name similarity is one input signal. When that signal is unavailable, the algorithm should still work. If it can't, the algorithm is too dependent on name similarity and needs to be stronger structurally.

## What makes cross-lingual structural matching work

### Multilingual relation embeddings

Relation variation is semantic: "acquired", "買収した", "a acheté" are different words expressing the same meaning. Multilingual sentence embedding models (LaBSE, multilingual E5, multilingual MiniLM) place these close together regardless of language.

Swapping the current English sentence model for a multilingual one extends relation similarity to cross-lingual matching with no architectural change. The propagation algorithm already uses continuous relation similarity as a gate — it doesn't care what language the phrases are in.

### Shared proper nouns as seeds

Most international news contains entity names that are partially or fully shared across scripts: "Apple", "NATO", "COVID-19", "Boeing 737". These provide nonzero name similarity even cross-lingually, seeding the propagation. The algorithm doesn't need all names to match — a few shared anchors are enough for structure to propagate the rest.

### Functionality across languages

Functionality computation pools edges with similar relation phrases. With multilingual embeddings, this pooling crosses language boundaries: "acquired" (English) and "買収した" (Japanese) pool together. This is correct — functionality reflects the semantic relation's properties, not its surface form.

## What needs to be strong enough

Cross-lingual matching is the stress test for the algorithm's structural capabilities. Two features directly address this:

**[Negative evidence](negative_evidence.md)** prevents false matches between entities that share some structural context but differ on specific functional relations. Without name similarity to help disambiguate, structural precision matters more — negative evidence provides it.

**[Progressive merging](progressive_merging.md)** enriches entity neighborhoods by combining matched entities during propagation. In the cross-lingual case, a match established through shared proper nouns in one part of the graph creates richer structure that can propagate matches to entities with no name overlap at all. Each committed merge makes the next match easier to find.

Both features make structural evidence more reliable, which is exactly what's needed when name evidence is zero.

## What changes for multilingual support

| Component | Change needed |
|-----------|--------------|
| Relation embeddings | Multilingual sentence model instead of English-only |
| Name similarity | None — works within-script, absent cross-script |
| Functionality | None — multilingual embeddings handle pooling |
| Propagation | None — language-agnostic by design |

The algorithm is language-agnostic. Language enters only through the embedding model choice.

## References

- Conneau, Lample, Ranzato, Denoyer, Jégou. *Word Translation Without Parallel Data.* ICLR 2018. (Background on cross-lingual embedding alignment; superseded for our use case by models pre-trained multilingually.)
