"""Embedder: persistent, cached text → unit-vector resource."""

from collections.abc import Callable

import numpy as np
from pydantic_ai import Embedder as PydanticAIEmbedder


class Embedder:
    """Wraps a pydantic-ai embedding model with text → unit-vector caching.

    The model is a provider-prefixed pydantic-ai string, e.g.
    ``sentence-transformers:Qwen/Qwen3-Embedding-0.6B``.
    """

    def __init__(self, model_name: str):
        self._embedder = PydanticAIEmbedder(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def embed(
        self,
        keys: list[str],
        template: Callable[[str], str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Embed texts, returning key → unit vector. Results are cached.

        If *template* is given, each key is transformed before encoding
        but the returned dict is still keyed by the original key.
        """
        texts = [template(k) for k in keys] if template else keys
        missing = [t for t in texts if t not in self._cache]
        if missing:
            result = self._embedder.embed_documents_sync(missing)
            for t, v in zip(missing, result.embeddings):
                vec = np.asarray(v)
                self._cache[t] = vec / np.linalg.norm(vec)
        return {k: self._cache[t] for k, t in zip(keys, texts)}
