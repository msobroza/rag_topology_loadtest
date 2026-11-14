"""domino/app.sh exports the OVMS models on a first start that finds none.

app.sh is copied into a tree of its own whose ``ovms/export_models.sh`` is a fake,
so the real exporter (pip, curl, the Hub) can never run. PATH holds only fakes
and links to the few system tools the script uses, so nothing installed on this
machine (``uv``, ``rag-served-models``, ``ovms``) is reached either.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

SYSTEM_TOOLS = ("bash", "env", "cat", "dirname", "mkdir", "chmod", "id")
EXPORTER_VENV = "ovms-exporter-venv"


def write_fake(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # never write through a link to a system tool
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def logging_fake(path: Path, log_file: Path) -> None:
    write_fake(path, f'echo "$@" >> "{log_file}"\n')


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """app.sh beside a fake export_models.sh that records argv, PYTHON and the
    ``rag-served-models`` it found, then does what FAKE_EXPORT says: ``writes``
    both configs, ``fails``, or ``writes-nothing``."""
    tree = tmp_path / "tree"
    (tree / "domino").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "domino" / "app.sh", tree / "domino" / "app.sh")
    (tree / ".python-version").write_text("3.11\n")
    write_fake(
        tree / "ovms" / "export_models.sh",
        f'echo "$* PYTHON=$PYTHON served=$(command -v rag-served-models)"'
        f' >> "{tmp_path}/export.log"\n'
        'case "$FAKE_EXPORT" in\n'
        '  writes) mkdir -p "$1"; echo "{}" > "$1/config_reranker.json"; '
        'echo "{}" > "$1/config_embedder.json" ;;\n'
        "  fails) exit 3 ;;\n"
        "esac\n",
    )
    return tree


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake ``ovms`` and ``uv``; ``uv venv <dir>`` fills <dir>/bin with fakes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in SYSTEM_TOOLS:
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bin_dir / name).symlink_to(found)
    logging_fake(bin_dir / "ovms", tmp_path / "ovms.log")
    venv_fakes = "".join(
        f"printf '#!/usr/bin/env bash\\n' > \"$2/bin/{name}\"; "
        f'chmod +x "$2/bin/{name}"\n'
        for name in ("rag-served-models", "python")
    )
    write_fake(
        bin_dir / "uv",
        f'echo "$@" >> "{tmp_path}/uv.log"\n'
        f'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n{venv_fakes}fi\n',
    )
    return bin_dir


def run_app(
    tmp_path: Path, tree: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    full_env = {"PATH": str(bin_dir), "HOME": str(home), **env}
    return subprocess.run(
        ["bash", str(tree / "domino" / "app.sh")],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def log(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / f"{name}.log"
    return path.read_text().splitlines() if path.exists() else []


def test_the_fake_stands_where_app_sh_runs_the_real_exporter() -> None:
    assert (REPO_ROOT / "ovms" / "export_models.sh").is_file()
    assert (
        'ovms/export_models.sh "$MODELS_DIR"'
        in (REPO_ROOT / "domino" / "app.sh").read_text()
    )


def test_a_present_config_is_served_without_any_export(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "config_reranker.json").write_text("{}")

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        RAG_ROLE="ovms-reranker",
        RAG_OVMS_MODELS_DIR=str(models),
    )

    assert result.returncode == 0, result.stderr
    assert (log(tmp_path, "export"), log(tmp_path, "uv")) == ([], [])
    assert log(tmp_path, "ovms") == [
        f"--rest_port 8888 --config_path {models}/config_reranker.json"
    ]


def test_a_missing_config_is_exported_with_a_separate_python_then_served(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    models = tmp_path / "models"

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        RAG_ROLE="ovms-embedder",
        RAG_OVMS_MODELS_DIR=str(models),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    home = tmp_path / "home"
    project, exporter = home / "rag-venv", home / EXPORTER_VENV
    assert log(tmp_path, "uv") == [
        f"venv {project} --python 3.11",  # the project, for rag-served-models
        f"pip install --python {project}/bin/python .",
        f"venv {exporter} --seed --python 3.11",  # the exporter's own pins go here
    ]
    assert log(tmp_path, "export") == [
        f"{models} PYTHON={exporter}/bin/python "
        f"served={project}/bin/rag-served-models"
    ]
    assert log(tmp_path, "ovms") == [
        f"--rest_port 8888 --config_path {models}/config_embedder.json"
    ]
    assert f"exporting the models into {models}" in result.stderr


def test_a_rag_served_models_already_on_path_is_used_as_is(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    write_fake(bin_dir / "rag-served-models", "")

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        RAG_ROLE="ovms-reranker",
        RAG_OVMS_MODELS_DIR=str(tmp_path / "models"),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    exporter = tmp_path / "home" / EXPORTER_VENV
    assert log(tmp_path, "uv") == [f"venv {exporter} --seed --python 3.11"]
    assert log(tmp_path, "export")[0].endswith(f"served={bin_dir}/rag-served-models")


def test_bootstrap_installs_the_ovms_bundle_then_exports(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "lib").mkdir(parents=True)
    logging_fake(bundle / "bin" / "ovms", tmp_path / "bundle-ovms.log")
    logging_fake(bin_dir / "apt-get", tmp_path / "apt-get.log")
    write_fake(bin_dir / "id", "echo 0\n")  # root: apt-get without sudo
    models = tmp_path / "models"

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        RAG_ROLE="ovms-reranker",
        RAG_BOOTSTRAP="1",
        RAG_OVMS_BUNDLE=str(bundle),
        RAG_OVMS_MODELS_DIR=str(models),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    assert len(log(tmp_path, "apt-get")) == 2
    assert len(log(tmp_path, "export")) == 1
    assert log(tmp_path, "bundle-ovms") == [
        f"--rest_port 8888 --config_path {models}/config_reranker.json"
    ]


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        ("fails", "exporting the OVMS models into"),
        ("writes-nothing", "is still missing"),
    ],
)
def test_an_export_that_yields_no_config_stops_the_role(
    tmp_path: Path, tree: Path, bin_dir: Path, outcome: str, message: str
) -> None:
    models = tmp_path / "models"

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        RAG_ROLE="ovms-reranker",
        RAG_OVMS_MODELS_DIR=str(models),
        FAKE_EXPORT=outcome,
    )

    assert result.returncode == 1
    assert "app.sh[ovms-reranker]: " in result.stderr
    assert message in result.stderr
    assert "ovms/export_models.sh" in result.stderr
    assert "RAG_OVMS_MODELS_DIR" in result.stderr
    assert f"{models}/config_reranker.json" in result.stderr
    assert log(tmp_path, "ovms") == []


def test_without_uv_a_missing_config_says_what_is_needed(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    (bin_dir / "uv").unlink()

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        RAG_ROLE="ovms-embedder",
        RAG_OVMS_MODELS_DIR=str(tmp_path / "models"),
    )

    assert result.returncode == 1
    assert "OVMS config not found" in result.stderr
    assert "'uv'" in result.stderr
    assert "RAG_OVMS_MODELS_DIR" in result.stderr
    assert log(tmp_path, "export") == []
