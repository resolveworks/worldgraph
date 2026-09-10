"""Project-wide constants."""

RELATION_TEMPLATE = "A {} B".format

# Minimum cosine similarity for two relation phrases (embedded) to be
# treated as equivalent — used for clustering, functionality pooling,
# and propagation gating.
RELATION_THRESHOLD = 0.8

# Confidence at which the pipeline commits an entity merge seeded by
# name similarity. Also used as the entity name-match cutoff in the
# extraction eval harness.
MERGE_THRESHOLD = 0.9
