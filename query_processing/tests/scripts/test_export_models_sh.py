"""ovms/export_models.sh asks ``qp-served-models``; fake binaries record argv here.

PATH is the fakes plus /usr/bin:/bin only, so neither a real ``curl``, a real
exporter Python nor an installed ``qp-served-models`` can be reached, and the
models directory is under ``tmp_path``: nothing touches the network or the repo.
Each embedder and reranker ``qp-served-models`` lists is exported under its
component's name, whatever the name; an LLM is skipped, for OVMS pulls it itself.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from repo_paths import TOOLKIT_ROOT

EXPORT_SH = TOOLKIT_ROOT / "ovms" / "export_models.sh"
SYSTEM_PATH = "/usr/bin:/bin"  # dirname, mkdir, env: what the script itself needs

# Quoted the way qp-served-models quotes, so the script's eval is exercised too:
# name, type, source id, alias, weight format (the same on every line). The
# reranker is named after no type.
FAKE_SERVED_MODELS = """\
embedder embedder org/custom-embed custom-embed fp32
coarse reranker org/custom-rerank 'custom rerank; touch pwned' fp32
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


def served_models_fake(bin_dir: Path, weight_format: str = "fp32") -> None:
    """A ``qp-served-models`` printing FAKE_SERVED_MODELS at ``weight_format``."""
    output = FAKE_SERVED_MODELS.replace(" fp32\n", f" {weight_format}\n")
    write_fake(bin_dir, "qp-served-models", f"cat <<'EOF'\n{output}EOF\n")


def calls(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / f"{name}.log"
    return path.read_text().splitlines() if path.exists() else []


def export_call(tmp_path: Path, task: str) -> str:
    (line,) = [c for c in calls(tmp_path, "python") if f"<{task}>" in c]
    return line


def test_each_served_model_is_exported_under_its_component_name(
    tmp_path: Path, bin_dir: Path
) -> None:
    served_models_fake(bin_dir)

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
    served_models_fake(bin_dir)

    result = run_export(tmp_path, bin_dir, with_folder=False)

    assert result.returncode == 0, result.stderr
    models = tmp_path / "ovms" / "models"
    embedder = export_call(tmp_path, "embeddings_ov")
    assert f"<--config_file_path><{models}/config_embedder.json>" in embedder
    assert (models / ".export").is_dir()


def test_an_llm_is_skipped_for_ovms_pulls_it_itself(
    tmp_path: Path, bin_dir: Path
) -> None:
    lines = "llm llm org/chat chat fp32\n" + FAKE_SERVED_MODELS
    write_fake(bin_dir, "qp-served-models", f"cat <<'EOF'\n{lines}EOF\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    exports = [c for c in calls(tmp_path, "python") if "<--source_model>" in c]
    assert len(exports) == 2  # the embedder and the reranker, not the LLM
    assert not any("org/chat" in c for c in exports)
    assert not (tmp_path / "models" / "config_llm.json").exists()


def test_a_workflow_file_of_an_llm_alone_installs_no_exporter(
    tmp_path: Path, bin_dir: Path
) -> None:
    # Nothing to export: neither the exporter's download nor its install runs.
    write_fake(
        bin_dir, "qp-served-models", "cat <<'EOF'\nllm llm org/chat chat fp32\nEOF\n"
    )

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    assert calls(tmp_path, "curl") == []
    assert calls(tmp_path, "python") == []
    assert "llm is an LLM, which OVMS pulls itself; skipped" in result.stdout


def test_a_served_model_of_an_unknown_type_stops_the_export(
    tmp_path: Path, bin_dir: Path
) -> None:
    lines = "tagger tagger org/tag tag fp32\n"
    write_fake(bin_dir, "qp-served-models", f"cat <<'EOF'\n{lines}EOF\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 1
    assert (
        "component 'tagger' is of type 'tagger'; expected an embedder, a reranker "
        "or an LLM"
    ) in result.stderr


def test_without_qp_served_models_the_export_fails_before_any_download(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_export(tmp_path, bin_dir)

    assert result.returncode != 0
    assert "qp-served-models" in result.stderr
    assert "make install" in result.stderr
    assert calls(tmp_path, "curl") == []
    assert calls(tmp_path, "python") == []


def test_path_hides_any_installed_qp_served_models() -> None:
    # Guards the failure test above: it is only meaningful if PATH has no real one.
    for directory in SYSTEM_PATH.split(os.pathsep):
        assert not (Path(directory) / "qp-served-models").exists()


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
    (models / "config_coarse.json").write_text("{}")
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


def test_a_qp_served_models_without_the_weight_format_says_to_reinstall(
    tmp_path: Path, bin_dir: Path
) -> None:
    # An older qp-served-models prints four words a line: no weight format.
    older = FAKE_SERVED_MODELS.replace(" fp32\n", "\n")
    write_fake(bin_dir, "qp-served-models", f"cat <<'EOF'\n{older}EOF\n")

    result = run_export(tmp_path, bin_dir)

    assert result.returncode == 1
    assert "printed no weight format" in result.stderr
    assert "make install" in result.stderr
    assert calls(tmp_path, "python") == []
