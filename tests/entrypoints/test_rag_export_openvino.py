"""``rag-export-openvino``: which models it exports, where, and how it fails.

The export functions are replaced by a recorder; their own behaviour is tested
in ``tests/adapters/test_openvino_export.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_load_test.entrypoints import rag_export_openvino
from rag_load_test.entrypoints.rag_export_openvino import main
from rag_load_test.infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
)

DEFAULT_EMBEDDER = "Snowflake/snowflake-arctic-embed-l-v2.0"
DEFAULT_RERANKER = "antoinelouis/crossencoder-camembert-base-mmarcoFR"


class RecordingExports:
    """Replaces both export functions; ``fail_with`` makes the next call raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path]] = []
        self.fail_with: Exception | None = None

    def embedder(self, model_id: str, target: Path) -> Path:
        return self._record("embedder", model_id, target)

    def reranker(self, model_id: str, target: Path) -> Path:
        return self._record("reranker", model_id, target)

    def _record(self, role: str, model_id: str, target: Path) -> Path:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append((role, model_id, target))
        return target


@pytest.fixture
def exports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RecordingExports:
    """Recorder in place, no RAG_* variable from the shell, no stray ``.env``."""
    recorder = RecordingExports()
    monkeypatch.setattr(rag_export_openvino, "export_embedder", recorder.embedder)
    monkeypatch.setattr(rag_export_openvino, "export_reranker", recorder.reranker)
    for name in [n for n in os.environ if n.startswith("RAG_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return recorder


def test_exports_both_settings_models_into_the_settings_folder(
    exports: RecordingExports, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([]) == 0

    base = Path("models/openvino")
    assert exports.calls == [
        ("embedder", DEFAULT_EMBEDDER, base / "snowflake-arctic-embed-l-v2.0"),
        ("reranker", DEFAULT_RERANKER, base / "crossencoder-camembert-base-mmarcoFR"),
    ]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"rag-export-openvino: embedder {DEFAULT_EMBEDDER} -> "
        f"{base / 'snowflake-arctic-embed-l-v2.0'}",
        f"rag-export-openvino: reranker {DEFAULT_RERANKER} -> "
        f"{base / 'crossencoder-camembert-base-mmarcoFR'}",
    ]


@pytest.mark.parametrize("role", ["embedder", "reranker"])
def test_only_exports_one_model(exports: RecordingExports, role: str) -> None:
    assert main(["--only", role]) == 0

    assert [call[0] for call in exports.calls] == [role]


def test_models_dir_option_beats_the_setting(
    exports: RecordingExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_OPENVINO_MODELS_DIR", "from-env")

    main(["--only", "reranker", "--models-dir", "from-flag"])

    assert exports.calls[0][2] == Path("from-flag/crossencoder-camembert-base-mmarcoFR")


def test_settings_overrides_pick_the_models_and_folder(
    exports: RecordingExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "org/embed")
    monkeypatch.setenv("RAG_RERANK_MODEL", "org/rerank")
    monkeypatch.setenv("RAG_OPENVINO_MODELS_DIR", "from-env")

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
    assert "models/openvino/snowflake-arctic-embed-l-v2.0" in err
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
        f"rag-export-openvino: cannot export embedder {DEFAULT_EMBEDDER} to "
        f"{Path('out/snowflake-arctic-embed-l-v2.0')}: {error}"
    ]
