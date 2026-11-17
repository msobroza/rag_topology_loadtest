"""``qp-export-openvino``: which models it exports, where, and how it fails.

The export functions are replaced by a recorder; their own behaviour is tested
in ``tests/adapters/test_openvino_export.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from query_processing.entrypoints import qp_export_openvino
from query_processing.entrypoints.qp_export_openvino import main
from query_processing.infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
)
from rag_components import shell_settings

DEFAULT_EMBEDDER = "fixture-org/fixture-embedder"
DEFAULT_RERANKER = "fixture-org/fixture-reranker"


class RecordingExports:
    """Replaces both export functions; ``fail_with`` makes the next call raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path]] = []
        self.weight_formats: list[str] = []
        self.fail_with: Exception | None = None

    def embedder(self, model_id: str, target: Path, weight_format: str) -> Path:
        return self._record("embedder", model_id, target, weight_format)

    def reranker(self, model_id: str, target: Path, weight_format: str) -> Path:
        return self._record("reranker", model_id, target, weight_format)

    def _record(
        self, role: str, model_id: str, target: Path, weight_format: str
    ) -> Path:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append((role, model_id, target))
        self.weight_formats.append(weight_format)
        return target


@pytest.fixture
def exports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RecordingExports:
    """Recorder in place, no QP_* variable from the shell, no stray ``.env``."""
    recorder = RecordingExports()
    monkeypatch.setattr(qp_export_openvino, "export_embedder", recorder.embedder)
    monkeypatch.setattr(qp_export_openvino, "export_reranker", recorder.reranker)
    for name in shell_settings(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return recorder


def test_exports_both_settings_models_into_the_settings_folder(
    exports: RecordingExports, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([]) == 0

    base = Path("models/openvino")
    assert exports.calls == [
        ("embedder", DEFAULT_EMBEDDER, base / "fixture-embedder"),
        ("reranker", DEFAULT_RERANKER, base / "fixture-reranker"),
    ]
    assert exports.weight_formats == ["fp32", "fp32"]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"qp-export-openvino: embedder {DEFAULT_EMBEDDER} (fp32) -> "
        f"{base / 'fixture-embedder'}",
        f"qp-export-openvino: reranker {DEFAULT_RERANKER} (fp32) -> "
        f"{base / 'fixture-reranker'}",
    ]


def test_an_int8_weight_format_exports_into_folders_of_their_own(
    exports: RecordingExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT", "int8")

    assert main(["--models-dir", "out"]) == 0

    assert [call[2] for call in exports.calls] == [
        Path("out/fixture-embedder-int8"),
        Path("out/fixture-reranker-int8"),
    ]
    assert exports.weight_formats == ["int8", "int8"]


@pytest.mark.parametrize("role", ["embedder", "reranker"])
def test_only_exports_one_model(exports: RecordingExports, role: str) -> None:
    assert main(["--only", role]) == 0

    assert [call[0] for call in exports.calls] == [role]


def test_a_component_of_any_name_is_exported_by_its_type_from_its_source_id(
    exports: RecordingExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__COARSE__TYPE", "reranker")
    monkeypatch.setenv("QP_COMPONENTS__COARSE__BACKEND", "local")
    monkeypatch.setenv("QP_COMPONENTS__COARSE__SOURCE_ID", "org/coarse-rr")

    assert main(["--only", "coarse"]) == 0

    assert exports.calls == [
        ("reranker", "org/coarse-rr", Path("models/openvino/coarse-rr"))
    ]


def test_models_dir_option_beats_the_setting(
    exports: RecordingExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_LOCAL_MODELS__OPENVINO_DIR", "from-env")

    main(["--only", "reranker", "--models-dir", "from-flag"])

    assert exports.calls[0][2] == Path("from-flag/fixture-reranker")


def test_settings_overrides_pick_the_models_and_folder(
    exports: RecordingExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__EMBEDDER__SOURCE_ID", "org/embed")
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__SOURCE_ID", "org/rerank")
    monkeypatch.setenv("QP_LOCAL_MODELS__OPENVINO_DIR", "from-env")

    main([])

    assert exports.calls == [
        ("embedder", "org/embed", Path("from-env/embed")),
        ("reranker", "org/rerank", Path("from-env/rerank")),
    ]


def test_an_unknown_only_value_is_a_usage_error(exports: RecordingExports) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--only", "llm"])

    assert exit_info.value.code == 2
    assert exports.calls == []


def test_missing_packages_name_the_extra_without_a_traceback(
    exports: RecordingExports, capsys: pytest.CaptureFixture[str]
) -> None:
    exports.fail_with = OpenVinoUnavailableError(
        "missing module 'openvino'; pip install 'sentence-transformers[openvino]'"
    )

    assert main([]) == 1

    err = capsys.readouterr().err
    assert "sentence-transformers[openvino]" in err
    assert DEFAULT_EMBEDDER in err
    assert "models/openvino/fixture-embedder" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("error", [OSError("repo not found"), ValueError("bad")])
def test_a_failed_export_names_model_and_folder_and_stops(
    exports: RecordingExports,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    exports.fail_with = error

    assert main(["--models-dir", "out"]) == 1

    assert exports.calls == []
    err = capsys.readouterr().err.splitlines()
    assert err == [
        f"qp-export-openvino: cannot export embedder {DEFAULT_EMBEDDER} to "
        f"{Path('out/fixture-embedder')}: {error}"
    ]
