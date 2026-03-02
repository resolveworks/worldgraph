# Functionality

Functionality is a weighting scheme from the knowledge graph alignment literature (PARIS, FLORA) that controls how much evidence a shared relation provides when deciding whether two entities are the same.

## The problem it solves

During similarity propagation, we examine pairs of edges across two graphs and ask: does this shared structure tell us anything about whether the connected entities match?

Not all relations carry equal evidence. If two entities both have an edge `located_in → London`, that tells us very little — thousands of entities are located in London. But if two entities both have an edge `has_ssn → 12345`, that's near-certain evidence of a match.

Functionality captures this difference. It measures how close a relation is to being a mathematical function — mapping each input to a unique output.

## Definitions

### Local functionality

The local functionality of relation `r` for a specific head entity `h` is:

```
fun(r, h) = 1 / |{t : r(h, t)}|
```

If `h` has exactly one target via `r`, local functionality is 1. If `h` maps to 3 targets via `r`, it's 1/3.

Example: `hasCapital` is globally functional (most countries have one capital), but `fun(hasCapital, South Africa) = 1/3` because South Africa has three capitals. Local functionality captures these exceptions.

### Global functionality

The global functionality of relation `r` is:

```
fun(r) = |{h : exists t, r(h,t)}| / |{(h,t) : r(h,t)}|
```

This equals `1 / avg_out_degree` of `r`. If every head entity using `r` has exactly one target, `fun(r) = 1`. If on average each head maps to 3 targets, `fun(r) = 1/3`.

This is the harmonic mean of the local functionalities (PARIS Appendix A, option 4).

### Inverse functionality

Both definitions have an inverse counterpart that looks at the relation from the other direction:

```
fun_inv(r) = |{t : exists h, r(h,t)}| / |{(h,t) : r(h,t)}|
```

This is `1 / avg_in_degree`. It measures how uniquely a target determines its source.

## Forward vs. inverse: why both matter

For any relation, functionality and inverse functionality are typically different. Which one matters depends on which direction evidence is flowing.

Consider `acquired`:
- **Forward functionality is low.** A single company can acquire many targets (Apple acquired Beats, Shazam, NeXT, ...). Knowing that two source entities match and both acquired *something* doesn't tell you much about whether those targets match.
- **Inverse functionality is high.** Each company is typically acquired by exactly one buyer. Knowing that two target entities match (Beats = Beats) and both were acquired by *someone* is strong evidence those someones are the same entity.

Consider `citizen_of`:
- **Forward functionality is low to medium.** Some people hold dual citizenship.
- **Inverse functionality is very low.** Millions of people are citizens of the same country.

Consider `has_ssn` (social security number):
- **Forward functionality is 1.** Each person has exactly one SSN.
- **Inverse functionality is 1.** Each SSN belongs to exactly one person.

The general rule: during propagation, when examining an edge pair `r(a, b)` and `r(a', b')`:
- To infer `b = b'` from `a = a'`, weight by **forward** functionality (`fun(r)`)
- To infer `a = a'` from `b = b'`, weight by **inverse** functionality (`fun_inv(r)`)

## Role in the alignment rule

PARIS (Suchanek et al. 2011, Equation 4) formulates entity equivalence as:

> If there exist `r`, `y`, `y'` such that `r(x, y)` and `r(x', y')` and `y = y'` and `fun_inv(r)` is high, then `x = x'`.

The probability of equivalence is weighted by the inverse functionality. This is the backward direction: matched targets plus high inverse functionality implies matched sources.

FLORA (Peng et al. 2025, Equation 1) extends this by requiring **both** global and local functionality as separate premises in the alignment rule, combined with `min` aggregation. This means a strong match on one premise cannot compensate for a weak one — if the relation is locally non-functional for the specific entity in question, the evidence is suppressed even if the relation is globally functional.

## Phrase pooling

Standard PARIS/FLORA assume a fixed relation vocabulary shared across knowledge graphs. We have free-text relation phrases that may express the same relation differently ("acquired", "buys", "purchased").

When computing functionality, we pool edges whose relation phrases are similar (above a cosine similarity threshold). All of "acquired", "buys", "purchased" contribute to the same degree statistics. This gives us accurate functionality estimates even when the same semantic relation appears under different surface forms.

## References

- Suchanek, Abiteboul, Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* VLDB 2011. Section 3 (functionality definition), Section 4.1 (use in alignment), Appendix A (design alternatives for global functionality).
- Peng, Bonald, Suchanek. *FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic.* 2025. Section 3 (functionality and local functionality definitions), Section 5.1-5.2 (use in alignment rules).
