"""Layer 1 tests for relation similarity via sentence embeddings."""

import numpy as np

from worldgraph.constants import RELATION_TEMPLATE


def test_synonym_relations_score_higher_than_unrelated(embedder):
    """cos('acquired', 'purchased') should exceed cos('acquired', 'located in').

    Catches embedding model or template breakage early, rather than surfacing
    as mysterious propagation failures.
    """
    embeddings = embedder.embed(
        ["acquired", "purchased", "located in"], template=RELATION_TEMPLATE
    )

    sim_synonyms = float(np.dot(embeddings["acquired"], embeddings["purchased"]))
    sim_unrelated = float(np.dot(embeddings["acquired"], embeddings["located in"]))

    assert sim_synonyms > sim_unrelated, (
        f"Expected cos('acquired','purchased') [{sim_synonyms:.3f}] > "
        f"cos('acquired','located in') [{sim_unrelated:.3f}]"
    )
