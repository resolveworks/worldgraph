import os

import numpy as np
import pytest
from dotenv import load_dotenv

from worldgraph.embed import Embedder

load_dotenv()


@pytest.fixture(scope="session")
def embedder():
    return Embedder(os.environ["EMBEDDING_MODEL"])


@pytest.fixture(scope="session")
def embed(embedder):
    """Callable that embeds text on demand (same API as before)."""

    def get(text: str | list[str]) -> np.ndarray | dict[str, np.ndarray]:
        if isinstance(text, str):
            return embedder.embed([text])[text]
        return embedder.embed(list(text))

    return get


@pytest.fixture(scope="session")
def embed_relation(embedder):
    def get(phrase: str) -> np.ndarray:
        return embedder.embed_relation(phrase)

    return get
