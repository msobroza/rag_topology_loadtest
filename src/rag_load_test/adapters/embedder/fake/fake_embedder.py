"""The ``fake`` embedder: a hashed bag of words, deterministic and dependency-free.

Texts sharing words get similar vectors, so retrieval over the synthetic corpus
stays meaningful in tests and LLM-free load runs.

Example:
    >>> await FakeEmbedder(dim=8).embed(["refund policy"])
    [[...]]
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from ....infrastructure.fake_text import tokenize
from ..base_embedder import BaseEmbedder


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _token_vector(token: str, dim: int) -> list[float]:
    digest = hashlib.sha256(token.encode()).digest()
    return _normalise([digest[i % len(digest)] / 127.5 - 1.0 for i in range(dim)])


class FakeEmbedder(BaseEmbedder):
    """Hashed bag-of-words embedder: texts sharing words get similar vectors.

    Args:
        dim: Length of every vector.
        model_name: The name ``ready`` reports.
    """

    def __init__(self, dim: int = 16, model_name: str = "fake-embedder") -> None:
        self.dim, self.model_name = dim, model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed one text as the normalised sum of its tokens' hashed vectors.

        Args:
            text: The text to embed; one with no token embeds as ``<empty>``.

        Returns:
            A unit vector of length ``dim``.
        """
        acc = [0.0] * self.dim
        for token in tokenize(text) or ["<empty>"]:
            for i, value in enumerate(_token_vector(token, self.dim)):
                acc[i] += value
        return _normalise(acc)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with ``embed_text``.

        Args:
            texts: The texts to embed.

        Returns:
            One unit vector per text, in the same order.
        """
        return [self.embed_text(t) for t in texts]

    async def ready(self) -> tuple[bool, str]:
        """Report ready; the fake has nothing to load.

        Returns:
            ``True`` and the model name.
        """
        return True, self.model_name
