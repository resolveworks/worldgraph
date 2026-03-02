import os

import numpy as np
import pytest
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()


@pytest.fixture(scope="session")
def embed():
    """Return a callable that embeds text on demand, caching results.

    Accepts a single string or a list of strings. Returns a single vector
    or a dict of {text: vector} respectively.
    """
    model = TextEmbedding(model_name=os.environ["EMBEDDING_MODEL"])
    cache: dict[str, np.ndarray] = {}

    def get(text: str | list[str]) -> np.ndarray | dict[str, np.ndarray]:
        if isinstance(text, str):
            if text not in cache:
                cache[text] = np.array(next(model.embed([text])))
            return cache[text]
        for t in text:
            if t not in cache:
                cache[t] = np.array(next(model.embed([t])))
        return {t: cache[t] for t in text}

    return get


@pytest.fixture(scope="session")
def embed_relation(embed):
    """Embed a relation phrase with syntactic context wrapping."""

    def get(phrase: str) -> np.ndarray:
        return embed(f"A {phrase} B")

    return get
