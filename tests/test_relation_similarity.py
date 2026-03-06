"""Layer 1 tests for relation phrase similarity via sentence embeddings."""

import numpy as np

from worldgraph.constants import RELATION_TEMPLATE


def test_synonym_phrases_rank_higher_than_unrelated(embedder):
    """Synonym phrases should be more similar to each other than to unrelated phrases.

    cos("acquired", "purchased") > cos("acquired", "located in")
    Catches embedding model or template breakage at the right layer.
    """
    phrases = ["acquired", "purchased", "located in"]
    embeddings = embedder.embed(phrases, template=RELATION_TEMPLATE)

    sim_synonyms = float(np.dot(embeddings["acquired"], embeddings["purchased"]))
    sim_unrelated = float(np.dot(embeddings["acquired"], embeddings["located in"]))

    assert sim_synonyms > sim_unrelated, (
        f"Expected cos('acquired','purchased') [{sim_synonyms:.3f}] > "
        f"cos('acquired','located in') [{sim_unrelated:.3f}]"
    )
