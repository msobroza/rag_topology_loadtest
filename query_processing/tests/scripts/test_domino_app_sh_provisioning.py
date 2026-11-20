"""An OVMS model server exports the OVMS models on a first start that finds none.

domino/app.sh and the launch scripts of the model-server kinds are copied into a
tree of their own whose ``ovms/export_models.sh`` is a fake, so the real exporter
(pip, curl, the Hub) can never run. PATH holds only fakes and links to the few
system tools the scripts use, so nothing installed on this machine (``uv``,
``qp-served-models``, ``ovms``) is reached either.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    model_server_env,
    served_config,
    wheels_env,
    write_ovms_config_stand_in,
)
from repo_paths import CORE_ROOT, TOOLKIT_ROOT

SYSTEM_TOOLS = ("bash", "env", "cat", "dirname", "mkdir", "chmod", "id", "mktemp")
EXPORTER_VENV = "ovms-exporter-venv"
OVMS_SH = TOOLKIT_ROOT / "domino" / "model_servers" / "ovms.sh"


def write_fake(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # never write through a link to a system tool
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def logging_fake(path: Path, log_file: Path) -> None:
    write_fake(path, f'echo "$@" >> "{log_file}"\n')


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """The launch scripts beside a fake export_models.sh that records argv, PYTHON
    and the ``qp-served-models`` it found, then does what FAKE_EXPORT says:
    ``writes`` the configs of the embedder, the reranker and a reranker named
    coarse, ``fails``, or ``writes-nothing``. Laid out as the repository is: the
    toolkit's directory, returned, with ``.python-version`` one level up."""
    repository = tmp_path / "repository"
    tree = repository / TOOLKIT_ROOT.name
    shutil.copytree(TOOLKIT_ROOT / "domino", tree / "domino")
    (repository / ".python-version").write_text("3.11\n")
    writes = "".join(
        f'echo "{{}}" > "$1/config_{name}.json"; '
        for name in ("reranker", "embedder", "coarse")
    )
    write_fake(
        tree / "ovms" / "export_models.sh",
        f'echo "$* PYTHON=$PYTHON served=$(command -v qp-served-models)"'
        f' >> "{tmp_path}/export.log"\n'
        f'pwd >> "{tmp_path}/export.cwd"\n'
        'case "$FAKE_EXPORT" in\n'
        f'  writes) mkdir -p "$1"; {writes};;\n'
        "  fails) exit 3 ;;\n"
        "esac\n",
    )
    return tree


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake ``ovms``, ``qp-ovms-config`` and ``uv``; ``uv venv <dir>`` fills
    <dir>/bin with fakes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in SYSTEM_TOOLS:
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bin_dir / name).symlink_to(found)
    logging_fake(bin_dir / "ovms", tmp_path / "ovms.log")
    write_ovms_config_stand_in(bin_dir / "qp-ovms-config", tmp_path)
    venv_fakes = "".join(
        f"printf '#!/usr/bin/env bash\\n' > \"$2/bin/{name}\"; "
        f'chmod +x "$2/bin/{name}"\n'
        for name in ("qp-served-models", "python")
    )
    write_fake(
        bin_dir / "uv",
        f'echo "$@" >> "{tmp_path}/uv.log"\n'
        f'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n{venv_fakes}fi\n',
    )
    return bin_dir


def run_app(
    tmp_path: Path, tree: Path, bin_dir: Path, component: str, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run the copied app.sh as an OVMS model server of ``component``.

    Its wheels' directory holds doclang's core, as a deployed App's does, so an
    install it makes to export finds it.
    """
    home, temporary = tmp_path / "home", tmp_path / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    full_env = {
        "PATH": str(bin_dir),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        **model_server_env("ovms", component),
        **wheels_env(tmp_path),
        **env,
    }
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


def test_the_fake_stands_where_the_ovms_kind_runs_the_real_exporter() -> None:
    assert (TOOLKIT_ROOT / "ovms" / "export_models.sh").is_file()
    assert '"$TOOLKIT/ovms/export_models.sh" "$OVMS_MODELS_DIR"' in OVMS_SH.read_text()


def test_a_present_config_is_served_without_any_export(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "config_reranker.json").write_text("{}")
    (models / "weight-format").write_text("fp32\n")

    result = run_app(
        tmp_path, tree, bin_dir, "reranker", QP_OVMS_MODELS_DIR=str(models)
    )

    assert result.returncode == 0, result.stderr
    assert (log(tmp_path, "export"), log(tmp_path, "uv")) == ([], [])
    assert log(tmp_path, "ovms") == [
        f"--rest_port 8888 --config_path {served_config(tmp_path)}"
    ]


@pytest.mark.parametrize("component", ["embedder", "coarse"])
def test_a_missing_config_is_exported_with_a_separate_python_then_served(
    tmp_path: Path, tree: Path, bin_dir: Path, component: str
) -> None:
    models = tmp_path / "models"

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        component,
        QP_OVMS_MODELS_DIR=str(models),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    home, found = tmp_path / "home", tmp_path / "wheels"
    project, exporter = home / "qp-venv", home / EXPORTER_VENV
    assert log(tmp_path, "uv") == [
        f"venv {project} --python 3.11",  # the project, for qp-served-models
        f"pip install --python {project}/bin/python --find-links {found} "
        f"{tree.parent / CORE_ROOT.name} {tree}",
        f"venv {exporter} --seed --python 3.11",  # the exporter's own pins go here
    ]
    assert log(tmp_path, "export") == [
        f"{models} PYTHON={exporter}/bin/python "
        f"served={project}/bin/qp-served-models"
    ]
    assert log(tmp_path, "ovms") == [
        f"--rest_port 8888 --config_path {served_config(tmp_path)}"
    ]
    assert f"exporting the models into {models}" in result.stderr


def test_the_export_runs_in_the_use_cases_directory_after_installing_both(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    # qp-served-models finds the use case's workflow file in its directory.
    use_case = tmp_path / "use_case"
    use_case.mkdir()

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        "reranker",
        QP_OVMS_MODELS_DIR=str(tmp_path / "models"),
        QP_USE_CASE_DIR=str(use_case),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    project, found = tmp_path / "home" / "qp-venv", tmp_path / "wheels"
    assert log(tmp_path, "uv")[1] == (
        f"pip install --python {project}/bin/python --find-links {found} "
        f"{tree.parent / CORE_ROOT.name} {tree} {use_case}"
    )
    assert (tmp_path / "export.cwd").read_text().splitlines() == [str(use_case)]


def test_a_qp_served_models_already_on_path_is_used_as_is(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    write_fake(bin_dir / "qp-served-models", "")

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        "reranker",
        QP_OVMS_MODELS_DIR=str(tmp_path / "models"),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    exporter = tmp_path / "home" / EXPORTER_VENV
    assert log(tmp_path, "uv") == [f"venv {exporter} --seed --python 3.11"]
    assert log(tmp_path, "export")[0].endswith(f"served={bin_dir}/qp-served-models")


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
        "reranker",
        QP_BOOTSTRAP="1",
        QP_OVMS_BUNDLE=str(bundle),
        QP_OVMS_MODELS_DIR=str(models),
        FAKE_EXPORT="writes",
    )

    assert result.returncode == 0, result.stderr
    assert len(log(tmp_path, "apt-get")) == 2
    assert len(log(tmp_path, "export")) == 1
    assert log(tmp_path, "bundle-ovms") == [
        f"--rest_port 8888 --config_path {served_config(tmp_path)}"
    ]


@pytest.mark.parametrize(
    ("component", "outcome", "message"),
    [
        ("reranker", "fails", "exporting the OVMS models into"),
        ("reranker", "writes-nothing", "is still missing"),
        # The exporter writes the configs of embedders and rerankers only.
        ("llm", "writes", "it exports each embedder and reranker"),
    ],
)
def test_an_export_that_yields_no_config_stops_the_model_server(
    tmp_path: Path,
    tree: Path,
    bin_dir: Path,
    component: str,
    outcome: str,
    message: str,
) -> None:
    models = tmp_path / "models"

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        component,
        QP_OVMS_MODELS_DIR=str(models),
        FAKE_EXPORT=outcome,
    )

    assert result.returncode == 1
    assert f"app.sh[model-server ovms {component}]: " in result.stderr
    assert message in result.stderr
    assert "ovms/export_models.sh" in result.stderr
    assert "QP_OVMS_MODELS_DIR" in result.stderr
    assert f"{models}/config_{component}.json" in result.stderr
    assert log(tmp_path, "ovms") == []


def test_without_uv_a_missing_config_says_what_is_needed(
    tmp_path: Path, tree: Path, bin_dir: Path
) -> None:
    (bin_dir / "uv").unlink()

    result = run_app(
        tmp_path,
        tree,
        bin_dir,
        "embedder",
        QP_OVMS_MODELS_DIR=str(tmp_path / "models"),
    )

    assert result.returncode == 1
    assert "OVMS config not found" in result.stderr
    assert "'uv'" in result.stderr
    assert "QP_OVMS_MODELS_DIR" in result.stderr
    assert log(tmp_path, "export") == []
