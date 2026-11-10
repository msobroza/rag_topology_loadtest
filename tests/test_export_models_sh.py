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

EXPORT_SH = Path(__file__).resolve().parents[1] / "ovms" / "export_models.sh"
SYSTEM_PATH = "/usr/bin:/bin"  # dirname, mkdir, env: what the script itself needs

# Quoted the way rag-served-models quotes, so the script's eval is exercised too.
FAKE_SERVED_MODELS = """\
RERANKER_SOURCE_MODEL=org/custom-rerank
RERANKER_MODEL_NAME='custom rerank'
EMBEDDER_SOURCE_MODEL=org/custom-embed
EMBEDDER_MODEL_NAME=custom-embed
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


def calls(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / f"{name}.log"
    return path.read_text().splitlines() if path.exists() else []


def export_call(tmp_path: Path, task: str) -> str:
    (line,) = [c for c in calls(tmp_path, "python") if f"<{task}>" in c]
    return line


def test_both_exports_get_the_served_models_from_rag_served_models(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_fake(bin_dir, "rag-served-models", f"cat <<'EOF'\n{FAKE_SERVED_MODELS}EOF\n")

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
    assert "rag-served-models" in result.stderr and "make install" in result.stderr
    assert calls(tmp_path, "curl") == [] and calls(tmp_path, "python") == []


def test_path_hides_any_installed_rag_served_models() -> None:
    # Guards the failure test above: it is only meaningful if PATH has no real one.
    for directory in SYSTEM_PATH.split(os.pathsep):
        assert not (Path(directory) / "rag-served-models").exists()
