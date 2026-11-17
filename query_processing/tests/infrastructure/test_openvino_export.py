"""The OpenVINO export over a fake sentence-transformers: no weights, no OpenVINO.

The real library loads the model with ``backend="openvino"`` (converting it in
memory) and ``save_pretrained`` writes a folder that loads back as is; the fake
records those calls and writes one marker file instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from query_processing.infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
    export_embedder,
    export_reranker,
)

OPENVINO_MODULES = ("openvino", "optimum", "optimum.intel", "optimum.intel.openvino")
# load_in_8bit is always passed: left out, optimum-intel picks by model size.
FP32_EXPORT = {"export": True, "load_in_8bit": False}
INT8_EXPORT = {"export": True, "load_in_8bit": True}


class FakeSentenceTransformersModule(ModuleType):
    """Stands in for ``sentence_transformers``; records loads and saves."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.SentenceTransformer = self._model_class("embedder")
        self.CrossEncoder = self._model_class("reranker")

    def _model_class(self, role: str) -> type:
        calls = self.calls

        class FakeModel:
            def __init__(self, name: str, **kwargs: Any) -> None:
                calls.append((role, name, kwargs))
                self.name = name

            def save_pretrained(self, path: str) -> None:
                calls.append((role, f"save:{Path(path).name}", {}))
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "openvino_model.xml").write_text(self.name)

        return FakeModel


@pytest.fixture
def fake_library(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name in OPENVINO_MODULES:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    return module


def test_export_embedder_loads_with_openvino_and_saves_the_folder(
    fake_library: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    target = tmp_path / "exports" / "embed"

    assert export_embedder("org/embed", target, "fp32") == target

    assert fake_library.calls[0] == (
        "embedder",
        "org/embed",
        {"device": "cpu", "backend": "openvino", "model_kwargs": FP32_EXPORT},
    )
    assert (target / "openvino_model.xml").read_text() == "org/embed"


def test_export_reranker_uses_the_cross_encoder(
    fake_library: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    target = tmp_path / "rerank"

    export_reranker("org/rerank", target, "fp32")

    assert fake_library.calls[0] == (
        "reranker",
        "org/rerank",
        {"device": "cpu", "backend": "openvino", "model_kwargs": FP32_EXPORT},
    )
    assert (target / "openvino_model.xml").read_text() == "org/rerank"


@pytest.mark.parametrize(
    ("weight_format", "model_kwargs"), [("fp32", FP32_EXPORT), ("int8", INT8_EXPORT)]
)
@pytest.mark.parametrize("export", [export_embedder, export_reranker])
def test_the_weight_format_says_whether_the_load_compresses_to_8_bits(
    fake_library: FakeSentenceTransformersModule,
    tmp_path: Path,
    export: Any,
    weight_format: str,
    model_kwargs: dict[str, bool],
) -> None:
    export("org/model", tmp_path / "model", weight_format)

    [(_role, _name, load)] = [c for c in fake_library.calls if c[1] == "org/model"]
    assert load["model_kwargs"] == model_kwargs


def test_export_overwrites_a_previous_export_cleanly(
    fake_library: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    target = tmp_path / "embed"
    target.mkdir()
    (target / "stale.bin").write_text("from an older export")

    export_embedder("org/embed", target, "fp32")

    assert sorted(p.name for p in target.iterdir()) == ["openvino_model.xml"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["embed"]


def test_a_failed_export_leaves_the_previous_folder_alone(
    fake_library: FakeSentenceTransformersModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "embed"
    target.mkdir()
    (target / "openvino_model.xml").write_text("previous")

    def refuse(self: Any, path: str) -> None:
        Path(path).mkdir(parents=True)
        raise OSError(f"disk full writing {path}")

    monkeypatch.setattr(fake_library.SentenceTransformer, "save_pretrained", refuse)

    with pytest.raises(OSError, match="disk full"):
        export_embedder("org/embed", target, "fp32")

    assert (target / "openvino_model.xml").read_text() == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["embed"]


@pytest.mark.parametrize("missing", ["openvino", "optimum.intel.openvino"])
def test_missing_openvino_packages_name_the_extra_before_any_load(
    fake_library: FakeSentenceTransformersModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setitem(sys.modules, missing, None)

    with pytest.raises(
        OpenVinoUnavailableError, match=r"sentence-transformers\[openvino\]"
    ):
        export_reranker("org/rerank", tmp_path / "rerank", "fp32")

    assert fake_library.calls == []
    assert not (tmp_path / "rerank").exists()
