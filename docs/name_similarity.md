# Name Similarity: Soft TF-IDF + Jaro-Winkler

Entity name matching uses Soft TF-IDF (Cohen, Ravikumar & Fienberg 2003), a hybrid string similarity metric that combines token-level IDF weighting with character-level Jaro-Winkler comparison. This document explains the algorithm, why it was chosen over embedding-based similarity, and our parameter choices.

## The problem

News articles refer to the same entity with surface-form variations:

- Truncation: "Meridian Technologies" / "Meridian Tech"
- Titles and honorifics: "Dr. Elena Vasquez" / "Elena Vasquez"
- Suffixes: "DataVault Inc" / "DataVault Incorporated"
- Abbreviations: "Corp" / "Corporation"

These are not semantic variations — "Meridian Tech" doesn't *mean* something similar to "Meridian Technologies", it *is* the same string with a truncated token. Embedding models treat them as separate phrases and may or may not place them close in vector space. String similarity handles them directly.

## Why not embeddings for names?

We originally used sentence embeddings for both name and relation similarity. Relations still use embeddings (see [similarity_flooding.md](similarity_flooding.md)) because relation variation *is* semantic: "acquired" and "purchased" are genuinely different words expressing the same meaning. But for entity names, embeddings have three problems:

1. **Inconsistent on surface-form variation.** Whether "Meridian Tech" and "Meridian Technologies" land close in embedding space depends on the model's training data. There's no guarantee, and the failure mode is silent.

2. **Hubness.** High-dimensional embedding spaces suffer from hubness — some points are nearest neighbors of many others regardless of true similarity. This creates false matches that are hard to debug.

3. **Overkill.** Loading a sentence embedding model to compare strings that share most of their characters is unnecessary. Soft TF-IDF with Jaro-Winkler is pure string comparison — no model, no GPU, deterministic.

## The algorithm

### IDF weights

First, compute IDF (inverse document frequency) for every token across all entity labels in the corpus:

```
IDF(token) = log(N / df(token))
```

where N is the total number of labels and df is how many labels contain the token. Common tokens like "Inc" or "Dr." get low weight; distinctive tokens like "Meridian" or "DataVault" get high weight.

### Jaro-Winkler inner similarity

Jaro-Winkler (Jaro 1989, Winkler 1999) is a character-level similarity metric designed for short strings like personal names. It counts common characters and transpositions, then boosts the score for matching prefixes. Key property: truncation-robust. "tech" vs "technologies" scores ~0.87 because they share a long common prefix and many common characters.

### Soft TF-IDF

Standard TF-IDF cosine similarity only counts exact token matches between two strings. Soft TF-IDF (Cohen et al. 2003) relaxes this: a token in string S can match a *similar* token in string T, where similarity is measured by Jaro-Winkler.

Given strings S and T, tokenized and normalized:

```
SoftTFIDF(S, T) = Σ_{w ∈ CLOSE(θ, S, T)}  V(w, S) · V(w', T) · D(w, T)
```

where:

- `CLOSE(θ, S, T)` = tokens w in S that have some token w' in T with `JW(w, w') >= θ`
- `D(w, T) = max_{w' ∈ T} JW(w, w')` — the best Jaro-Winkler match for w in T
- `V(w, X)` = L2-normalized IDF weight of token w in string X: `IDF(w) / sqrt(Σ IDF(w')²)` over all tokens w' in X

The L2 normalization means the score accounts for unmatched tokens on both sides. If S has extra tokens that don't match anything in T, they increase `||V_S||` (the denominator of `V(w, S)`), reducing each matched token's contribution. Same for extra tokens in T.

### Worked example

Comparing "Meridian Tech" vs "Meridian Technologies" with a corpus of ["Meridian Technologies", "Meridian Tech", "DataVault"]:

1. **Tokenize:** S = ["meridian", "tech"], T = ["meridian", "technologies"]
2. **IDF:** "meridian" appears in 2/3 labels → IDF = log(3/2) ≈ 0.41. "tech" in 1/3 → IDF = log(3) ≈ 1.10. "technologies" in 1/3 → IDF = log(3) ≈ 1.10.
3. **L2 norms:** ||S|| = sqrt(0.41² + 1.10²) ≈ 1.17. ||T|| = sqrt(0.41² + 1.10²) ≈ 1.17.
4. **Token matching:** "meridian" → "meridian" (JW=1.0, ≥ 0.85 ✓). "tech" → "technologies" (JW ≈ 0.87, ≥ 0.85 ✓).
5. **Score:** V("meridian", S) · V("meridian", T) · 1.0 + V("tech", S) · V("technologies", T) · 0.87 ≈ (0.41/1.17)² + (1.10/1.17)² · 0.87 ≈ 0.12 + 0.77 = **0.89**

The distinctive tokens ("tech"/"technologies") contribute most of the score because of their high IDF weight. The common prefix "meridian" contributes less. This is correct — the discriminative part of the name is what should matter.

## Parameter choices

### JW threshold: θ = 0.85

Cohen et al. use θ = 0.9 in their experiments. We lower this to 0.85. The reason: their benchmark datasets (census records, structured databases) have variation primarily from typos. News entity names have variation primarily from prefix truncation — and those token pairs land in the 0.85–0.90 range:

| Token pair | JW score |
|---|---|
| tech / technologies | 0.867 |
| corp / corporation | 0.873 |
| univ / university | 0.880 |
| dept / department | 0.860 |
| sys / systems | 0.867 |

At θ = 0.9, all of these are excluded. At θ = 0.85, they pass.

The false positive risk at 0.85 is manageable: token pairs like "data"/"date" (JW ≈ 0.88) could pass, but their impact is limited by IDF weighting and the structural matching layer above.

### What Soft TF-IDF does not handle

Some name variations are beyond any Jaro-Winkler threshold:

- **Single-letter initials:** "P." vs "Priya" (JW ≈ 0.76)
- **Consonant-cluster abbreviations:** "Dr." vs "Doctor" (JW ≈ 0.56), "Intl" vs "International" (JW ≈ 0.66)
- **Acronyms:** "FTC" vs "Federal Trade Commission"

These require structural evidence from similarity propagation (if the entities' neighborhoods match, they match despite low name similarity) or dedicated abbreviation handling.

## Unicode normalization

Input labels are NFKD-normalized with combining marks stripped before tokenization. This ensures "José García" and "Jose Garcia" are treated as identical — the accented characters decompose to base + combining mark, and the combining marks are removed.

## References

- Cohen, Ravikumar, Fienberg. *A Comparison of String Distance Metrics for Name-Matching Tasks.* IJCAI-03 Workshop on Information Integration on the Web, 2003.
- Jaro. *Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida.* JASA, 1989.
- Winkler. *The State of Record Linkage and Current Research Problems.* Statistics of Income Division, IRS, 1999.
