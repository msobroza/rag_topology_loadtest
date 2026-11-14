"""Doubles shared by the tests of the ``local`` embedder and reranker.

A scope that records where a call ran, scored passages, a stand-in for the
``sentence_transformers`` package that records each model's load, and one whose
models refuse overlapping calls as an OpenVINO model does. The fixtures over
them are in ``local_model_fixtures.py``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np

from rag_load_test.domain.passages import ScoredPassage
from rag_load_test.infrastructure.local_models.local_model import LocalModel
from rag_load_test.infrastructure.local_models.native_threads import NativeThreadScope


class RecordingScope(NativeThreadScope):
    """A real scope that also records the thread each call ran on."""

    def __init__(self, threads: int) -> None:
        super().__init__(threads)
        self.ran_on: list[str] = []

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        self.ran_on.append(threading.current_thread().name)
        return super().run(fn, *args)


def scored_passages(*ids: str) -> list[ScoredPassage]:
    return [ScoredPassage(id=i, text=f"text of {i}", retrieval_score=0.5) for i in ids]


class FakeLoadingModule(ModuleType):
    """Stands in for ``sentence_transformers``; records each model's load arguments."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.loads: list[tuple[str, str, dict[str, Any]]] = []
        loads = self.loads

        class SentenceTransformer:
            def __init__(self, source: str, **kwargs: Any) -> None:
                loads.append(("SentenceTransformer", source, kwargs))

        class CrossEncoder:
            def __init__(self, source: str, **kwargs: Any) -> None:
                loads.append(("CrossEncoder", source, kwargs))

        self.SentenceTransformer = SentenceTransformer
        self.CrossEncoder = CrossEncoder


OPENVINO_EXPORT = LocalModel("org/name", "models/openvino/name", "openvino")
OPENVINO_ARGUMENTS = {
    "device": "cpu",
    "backend": "openvino",
    "model_kwargs": {"ov_config": {"INFERENCE_NUM_THREADS": "3"}},
}


# Each runtime's load arguments, for ``from_pretrained``.
RUNTIME_LOADS = [
    (LocalModel("org/name", "org/name", "torch"), {"device": "cpu"}),
    (OPENVINO_EXPORT, OPENVINO_ARGUMENTS),
]


class FakeOneRequestModel:
    """A model with one inference request, as OpenVINO loads it.

    A call entered while another call on the same model is still running raises
    what OpenVINO raises; each call holds the request for ``CALL_SECONDS`` so
    calls that run at once do overlap.
    """

    CALL_SECONDS = 0.03

    def __init__(self, source: str, device: str, **runtime: object) -> None:
        del source, device, runtime  # passed as to the real classes
        self._request = threading.Lock()

    def encode(
        self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
    ) -> np.ndarray:
        self._infer()
        return np.array([[1.0, 0.0] for _ in texts])

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        self._infer()
        return [0.5 for _ in pairs]

    def _infer(self) -> None:
        if not self._request.acquire(blocking=False):
            raise RuntimeError("Infer Request is busy")
        try:
            time.sleep(self.CALL_SECONDS)
        finally:
            self._request.release()


class FakeOneRequestModule(ModuleType):
    """Stands in for ``sentence_transformers``; both models are ``FakeOneRequestModel``."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.SentenceTransformer = FakeOneRequestModel
        self.CrossEncoder = FakeOneRequestModel
