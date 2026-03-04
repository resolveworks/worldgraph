"""Embedder: persistent, cached text → unit-vector resource."""

from collections.abc import Callable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model with text → unit-vector caching."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
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
            vecs = self.model.encode(missing, normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return {k: self._cache[t] for k, t in zip(keys, texts)}
