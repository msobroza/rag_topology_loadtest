"""ovms/export_models.sh asks ``rag-served-models``; fake binaries record argv here.

PATH is the fakes plus /usr/bin:/bin only, so neither a real ``curl``, a real
exporter Python nor an installed ``rag-served-models`` can be reached, and the
models directory is under ``tmp_path``: nothing touches the network or the repo.
Each embedder and reranker ``rag-served-models`` lists is exported under its
component's name, whatever the name.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from repo_paths import TOOLKIT_ROOT

EXPORT_SH = TOOLKIT_ROOT / "ovms" / "export_models.sh"
SYSTEM_PATH = "/usr/bin:/bin"  # dirname, mkdir, env: what the script itself needs

# Quoted the way rag-served-models quotes, so the script's eval is exercised too:
# name, type, source id, alias. The reranker is named after no type.
FAKE_SERVED_MODELS = """\
embedder embedder org/custom-embed custom-embed
coarse reranker org/custom-rerank 'custom rerank; touch pwned'
"""


def write_fake(bin_dir: Path, name: str, body: str) -> None:
    script = bin_dir / name
    script.write_text("#!/usr/bin/env bash\n" + body)
    script.chmod(0o755)


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake ``python`` and ``curl`` that log each call as ``<arg><arg>...``.

    The fake ``python`` also reads its standard input to the end, as a real
    program may: it must not swallow the models still to export.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "curl"):
        write_fake(
            bin_dir,
            name,
            f"printf '<%s>' \"$@\" >> '{tmp_path}/{name}.log'\n"
            f"echo >> '{tmp_path}/{name}.log'\n"
            + ("cat > /dev/null\n" if name == "python" else ""),
        )
    return bin_dir


def run_export(
    tmp_path: Path, bin_dir: Path, *, with_folder: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run the script in ``tmp_path``, into tmp_path/models unless told no folder."""
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:{SYSTEM_PATH}",
        "PYTHON": str(bin_dir / "python"),
    }
    folder = [str(tmp_path / "models")] if with_folder else []
    return subprocess.run(
        ["bash", str(EXPORT_SH), *folder],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )


def calls(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / f"{name}.log"
    return path.read_text().splitlines() if path.exists() else []


def export_call(tmp_path: Path, task: str) -> str:
    (line,) = [c for c in calls(tmp_path, "python") if f"<{task}>" in c]
    return line


def test_each_served_model_is_exported_under_its_component_name(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_fake(bin_dir, "rag-served-models", f"cat <<'EOF'\n{FAKE_SERVED_MODELS}EOF\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    models = tmp_path / "models"
    reranker = export_call(tmp_path, "rerank_ov")
    assert "<--source_model><org/custom-rerank>" in reranker
    assert "<--model_name><custom rerank; touch pwned>" in reranker
    assert f"<--config_file_path><{models}/config_coarse.json>" in reranker
    embedder = export_call(tmp_path, "embeddings_ov")
    assert "<--source_model><org/custom-embed>" in embedder
    assert "<--model_name><custom-embed>" in embedder
    assert f"<--config_file_path><{models}/config_embedder.json>" in embedder
    assert not (tmp_path / "pwned").exists()  # a value is never run


def test_without_a_folder_the_models_land_in_the_working_directorys_ovms_models(
    tmp_path: Path, bin_dir: Path
) -> None:
    # The working directory is the use case's: every output of a command is there.
    write_fake(bin_dir, "rag-served-models", f"cat <<'EOF'\n{FAKE_SERVED_MODELS}EOF\n")

    result = run_export(tmp_path, bin_dir, with_folder=False)

    assert result.returncode == 0, result.stderr
    models = tmp_path / "ovms" / "models"
    embedder = export_call(tmp_path, "embeddings_ov")
    assert f"<--config_file_path><{models}/config_embedder.json>" in embedder
    assert (models / ".export").is_dir()


def test_a_served_model_of_another_type_stops_the_export(
    tmp_path: Path, bin_dir: Path
) -> None:
    lines = "llm llm org/chat chat\n"
    write_fake(bin_dir, "rag-served-models", f"cat <<'EOF'\n{lines}EOF\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 1
    assert "component 'llm' is of type 'llm'; expected an embedder or a reranker" in (
        result.stderr
    )


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
