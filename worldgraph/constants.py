"""Project-wide constants."""

# Confidence at which propagation commits a merge, gated on structural evidence.
MERGE_THRESHOLD = 0.7

# Neutral point of the evidence rule: a counterpart pair sitting exactly at
# this confidence is untested — it contributes neither positive nor negative
# evidence and does not count toward the tested-neighbor total. Also the
# default prior for statement pairs when no predicate-similarity prior is
# injected: predicates are display text and are never compared by the matcher.
NEUTRAL_PRIOR = 0.5
