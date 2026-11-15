"""ovms/export_models.sh asks ``rag-served-models``; fake binaries record argv here.

PATH is the fakes plus /usr/bin:/bin only, so neither a real ``curl``, a real
exporter Python nor an installed ``rag-served-models`` can be reached, and the
models directory is under ``tmp_path``: nothing touches the network or the repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

EXPORT_SH = REPO_ROOT / "ovms" / "export_models.sh"
SYSTEM_PATH = "/usr/bin:/bin"  # dirname, mkdir, env: what the script itself needs

# Quoted the way rag-served-models quotes, so the script's eval is exercised too.
FAKE_SERVED_MODELS = """\
RERANKER_SOURCE_MODEL=org/custom-rerank
RERANKER_MODEL_NAME='custom rerank'
EMBEDDER_SOURCE_MODEL=org/custom-embed
EMBEDDER_MODEL_NAME=custom-embed
OPENVINO_WEIGHT_FORMAT=fp32
"""


def write_fake(bin_dir: Path, name: str, body: str) -> None:
    script = bin_dir / name
    script.write_text("#!/usr/bin/env bash\n" + body)
    script.chmod(0o755)


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake ``python`` and ``curl`` that log each call as ``<arg><arg>...``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "curl"):
        write_fake(
            bin_dir,
            name,
            f"printf '<%s>' \"$@\" >> '{tmp_path}/{name}.log'\n"
            f"echo >> '{tmp_path}/{name}.log'\n",
        )
    return bin_dir


def run_export(tmp_path: Path, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:{SYSTEM_PATH}",
        "PYTHON": str(bin_dir / "python"),
    }
    return subprocess.run(
        ["bash", str(EXPORT_SH), str(tmp_path / "models")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def served_models_fake(bin_dir: Path, weight_format: str = "fp32") -> None:
    """A ``rag-served-models`` printing FAKE_SERVED_MODELS at ``weight_format``."""
    output = FAKE_SERVED_MODELS.replace(
        "OPENVINO_WEIGHT_FORMAT=fp32", f"OPENVINO_WEIGHT_FORMAT={weight_format}"
    )
    write_fake(bin_dir, "rag-served-models", f"cat <<'EOF'\n{output}EOF\n")


def calls(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / f"{name}.log"
    return path.read_text().splitlines() if path.exists() else []


def export_call(tmp_path: Path, task: str) -> str:
    (line,) = [c for c in calls(tmp_path, "python") if f"<{task}>" in c]
    return line


def test_both_exports_get_the_served_models_from_rag_served_models(
    tmp_path: Path, bin_dir: Path
) -> None:
    served_models_fake(bin_dir)

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    reranker = export_call(tmp_path, "rerank_ov")
    assert "<--source_model><org/custom-rerank>" in reranker
    assert "<--model_name><custom rerank>" in reranker
    embedder = export_call(tmp_path, "embeddings_ov")
    assert "<--source_model><org/custom-embed>" in embedder
    assert "<--model_name><custom-embed>" in embedder


def test_without_rag_served_models_the_export_fails_before_any_download(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_export(tmp_path, bin_dir)

    assert result.returncode != 0
    assert "rag-served-models" in result.stderr
    assert "make install" in result.stderr
    assert calls(tmp_path, "curl") == []
    assert calls(tmp_path, "python") == []


def test_path_hides_any_installed_rag_served_models() -> None:
    # Guards the failure test above: it is only meaningful if PATH has no real one.
    for directory in SYSTEM_PATH.split(os.pathsep):
        assert not (Path(directory) / "rag-served-models").exists()


@pytest.mark.parametrize("weight_format", ["fp32", "int8"])
def test_both_exports_use_the_weight_format_and_the_folder_records_it(
    tmp_path: Path, bin_dir: Path, weight_format: str
) -> None:
    served_models_fake(bin_dir, weight_format)

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    for task in ("rerank_ov", "embeddings_ov"):
        assert f"<--weight-format><{weight_format}>" in export_call(tmp_path, task)
    record = tmp_path / "models" / "weight-format"
    assert record.read_text() == f"{weight_format}\n"


def test_no_weight_format_is_hard_coded() -> None:
    assert "--weight-format int8" not in EXPORT_SH.read_text()


def test_a_folder_of_the_same_weight_format_is_exported_again(
    tmp_path: Path, bin_dir: Path
) -> None:
    served_models_fake(bin_dir, "int8")
    models = tmp_path / "models"
    models.mkdir()
    (models / "config_reranker.json").write_text("{}")
    (models / "weight-format").write_text("int8\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    assert len(calls(tmp_path, "python")) == 3  # pip, then both exports


@pytest.mark.parametrize(
    ("record", "found", "asked"),
    [("int8\n", "int8", "fp32"), ("fp32\n", "fp32", "int8"), (None, "int8", "fp32")],
    ids=["int8-folder", "fp32-folder", "folder-from-before-the-record"],
)
def test_a_folder_of_another_weight_format_is_refused_before_any_download(
    tmp_path: Path, bin_dir: Path, record: str | None, found: str, asked: str
) -> None:
    # The exporter skips a model folder that exists: it would keep the old weights.
    served_models_fake(bin_dir, asked)
    models = tmp_path / "models"
    models.mkdir()
    (models / "config_embedder.json").write_text("{}")
    if record is not None:
        (models / "weight-format").write_text(record)

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 1
    assert str(models) in result.stderr
    assert f"weight format {found}" in result.stderr
    assert f"{asked} was asked" in result.stderr
    assert "another folder" in result.stderr
    assert (calls(tmp_path, "curl"), calls(tmp_path, "python")) == ([], [])
    assert (models / "weight-format").exists() == (record is not None)


def test_a_rag_served_models_without_the_weight_format_says_to_reinstall(
    tmp_path: Path, bin_dir: Path
) -> None:
    older = FAKE_SERVED_MODELS.replace("OPENVINO_WEIGHT_FORMAT=fp32\n", "")
    write_fake(bin_dir, "rag-served-models", f"cat <<'EOF'\n{older}EOF\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 1
    assert "OPENVINO_WEIGHT_FORMAT" in result.stderr
    assert "make install" in result.stderr
    assert calls(tmp_path, "python") == []
