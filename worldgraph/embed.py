"""Embedder: persistent, cached text → unit-vector resource."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model with text → unit-vector caching."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Embed texts, returning text → unit vector. Results are cached."""
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return {t: self._cache[t] for t in texts}

    def embed_relation(self, phrase: str) -> np.ndarray:
        """Embed a relation phrase with syntactic context wrapping."""
        return self.embed([f"A {phrase} B"])[f"A {phrase} B"]
