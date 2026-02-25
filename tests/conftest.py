import numpy as np
import pytest
from fastembed import TextEmbedding


@pytest.fixture(scope="session")
def embed_phrase():
    """Return a callable that embeds a relation phrase on demand, caching results.

    Each unique phrase is embedded at most once per session.
    """
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    cache: dict[str, np.ndarray] = {}

    def get(phrase: str) -> np.ndarray:
        if phrase not in cache:
            cache[phrase] = np.array(next(model.embed([f"A {phrase} B"])))
        return cache[phrase]

    return get
