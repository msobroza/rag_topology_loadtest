"""A named fake for the ``sentence_transformers`` package: no weights, no torch."""

from __future__ import annotations

from types import ModuleType

import numpy as np


class FakeSentenceTransformersModule(ModuleType):
    """Stands in for the ``sentence_transformers`` package; records which weights load."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.loaded: list[str] = []
        self.batch_sizes: list[tuple[str, int]] = []
        self.load_arguments: list[dict[str, object]] = []  # the runtime's, per load
        loaded, batch_sizes = self.loaded, self.batch_sizes
        load_arguments = self.load_arguments

        class SentenceTransformer:
            def __init__(self, name: str, device: str, **runtime: object) -> None:
                del device  # passed by keyword, as to the real class
                loaded.append(f"embedder:{name}")
                load_arguments.append(runtime)

            def encode(
                self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
            ) -> np.ndarray:
                batch_sizes.append(("encode", batch_size))
                return np.array([[1.0, 0.0] for _ in texts])

        class CrossEncoder:
            def __init__(self, name: str, device: str, **runtime: object) -> None:
                del device  # passed by keyword, as to the real class
                loaded.append(f"reranker:{name}")
                load_arguments.append(runtime)

            def predict(
                self, pairs: list[tuple[str, str]], batch_size: int
            ) -> list[float]:
                batch_sizes.append(("predict", batch_size))
                return [0.5 for _ in pairs]

        self.SentenceTransformer = SentenceTransformer
        self.CrossEncoder = CrossEncoder
