"""A named fake for the ``sentence_transformers`` package: no weights, no torch.

It also stands in for what an OpenVINO export does with the package: the export
loads the model, then ``save_pretrained`` writes a folder (here, one marker file).
``fake_openvino_modules`` are the modules the export checks for before it starts.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import numpy as np

OPENVINO_MODULES = ("openvino", "optimum", "optimum.intel", "optimum.intel.openvino")


def fake_openvino_modules() -> dict[str, ModuleType]:
    """Empty modules under the names ``require_openvino`` imports; none installed."""
    return {name: ModuleType(name) for name in OPENVINO_MODULES}


class FakeSentenceTransformersModule(ModuleType):
    """Stands in for the ``sentence_transformers`` package; records which weights load.

    ``loaded`` holds ``<role>:<name>`` per constructor call, the export's load
    included; ``saved`` the folders ``save_pretrained`` wrote, ``save_threads`` the
    thread each save ran on. Set ``save_error`` to make ``save_pretrained`` write a
    partial file and then raise it.
    """

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.loaded: list[str] = []
        self.saved: list[str] = []
        self.save_threads: list[str] = []
        self.batch_sizes: list[tuple[str, int]] = []
        self.load_arguments: list[dict[str, object]] = []  # the runtime's, per load
        self.save_error: Exception | None = None
        self.SentenceTransformer = self._model_class("embedder")
        self.CrossEncoder = self._model_class("reranker")

    def _record_load(self, role: str, name: str, runtime: dict[str, object]) -> None:
        self.loaded.append(f"{role}:{name}")
        self.load_arguments.append(runtime)

    def _save(self, name: str, path: str) -> None:
        self.save_threads.append(threading.current_thread().name)
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "openvino_model.xml").write_text(name)
        if self.save_error is not None:
            raise self.save_error
        self.saved.append(path)

    def _model_class(self, role: str) -> type:
        fake = self

        class FakeModel:
            def __init__(self, name: str, device: str, **runtime: object) -> None:
                del device  # passed by keyword, as to the real class
                self.name = name
                fake._record_load(role, name, runtime)

            def save_pretrained(self, path: str) -> None:
                fake._save(self.name, path)

            def encode(
                self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
            ) -> np.ndarray:
                fake.batch_sizes.append(("encode", batch_size))
                return np.array([[1.0, 0.0] for _ in texts])

            def predict(
                self, pairs: list[tuple[str, str]], batch_size: int
            ) -> list[float]:
                fake.batch_sizes.append(("predict", batch_size))
                return [0.5 for _ in pairs]

        return FakeModel
