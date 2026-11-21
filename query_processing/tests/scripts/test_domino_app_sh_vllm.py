"""domino/app.sh as a model server of kind vllm: vLLM fetches and serves one model.

The script exports what ``qp-vllm-config --environment`` prints, then starts
``vllm serve`` on the words of ``qp-vllm-config --serve``. Here that command is
the toolkit's own, reading the tests' workflow file and the ``QP_VLLM_*``
variables; ``vllm``, ``uv`` and ``ldd`` are the stand-ins of
tests/helpers/launch_script_stand_ins.py, which record what would have run.
With QP_BOOTSTRAP=1 an environment without ``vllm`` gets it from the toolkit's
``vllm`` dependency group, in a virtualenv of its own. Nothing is installed and
no network is reached.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    model_server_env,
    put_the_real_toolkit_command_on_path,
    stand_in_env,
    started_env,
    started_in,
    vllm_argv,
    wheels_env,
    write_fake,
    write_vllm_stand_in,
)
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from repo_paths import CORE_ROOT, TOOLKIT_ROOT

VLLM_SH = TOOLKIT_ROOT / "domino" / "model_servers" / "vllm.sh"
EIGHT_GIB = str(8 * 2**30)
LISTENS = ["--host", "0.0.0.0", "--port", "8888"]
# The served words of each component of the tests' workflow file; the LLM's
# source id is given by its variable, as the file names none.
SERVED = {
    "embedder": [
        "fixture-org/fixture-embedder",
        "--served-model-name",
        "fixture-embedder",
    ],
    "reranker": [
        "fixture-org/fixture-reranker",
        "--served-model-name",
        "fixture-reranker",
    ],
    "llm": ["fixture-org/fixture-chat", "--served-model-name", "fixture-chat"],
}
LLM_SOURCE = {"QP_COMPONENTS__LLM__SOURCE_ID": "fixture-org/fixture-chat"}
GLIBC_FOUND = "ldd (Ubuntu GLIBC 2.35-0ubuntu3.8) 2.35"


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """The stand-ins, with the toolkit's real ``qp-vllm-config``; no ``vllm``."""
    bin_dir = make_stand_ins(tmp_path)
    put_the_real_toolkit_command_on_path(bin_dir, "qp-vllm-config")
    return bin_dir


def run_app(
    tmp_path: Path, path: Path, component: str = "embedder", **env: str
) -> subprocess.CompletedProcess[str]:
    """Run app.sh as the vllm model server of ``component``, ``path`` first on PATH."""
    workflow = {WORKFLOW_FILE_VARIABLE: os.environ[WORKFLOW_FILE_VARIABLE]}
    full = stand_in_env(
        tmp_path, path, **model_server_env("vllm", component), **workflow, **env
    )
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=full,
        cwd=tmp_path,  # no .env of the developer's is read
        capture_output=True,
        text=True,
        timeout=60,
    )


def bare_path(tmp_path: Path, *tools: str) -> Path:
    """A PATH folder with only ``tools`` linked from this machine, no uv, no qp-*."""
    bare = tmp_path / "bare-bin"
    bare.mkdir()
    for name in ("bash", "env", "dirname", "cat", *tools):
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bare / name).symlink_to(found)
    return bare


def last_message(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip().splitlines()[-1]


@pytest.mark.parametrize("component", list(SERVED))
def test_a_default_vllm_app_runs_the_one_default_command_line(
    tmp_path: Path, bin_dir: Path, component: str
) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(tmp_path, bin_dir, component, **LLM_SOURCE)

    assert result.returncode == 0, result.stderr
    assert vllm_argv(tmp_path) == [
        "serve",
        *SERVED[component],
        "--kv-cache-memory-bytes",
        EIGHT_GIB,
        *LISTENS,
    ]
    assert (log(tmp_path, "uv"), log(tmp_path, "nginx")) == ("", "")
    # No CPU variable is exported at the defaults.
    assert not [v for v in started_env(tmp_path, "vllm") if v.startswith("VLLM_")]


def test_a_named_setting_reaches_vllm(tmp_path: Path, bin_dir: Path) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(tmp_path, bin_dir, QP_VLLM_MAX_MODEL_LEN="4096")

    assert result.returncode == 0, result.stderr
    argv = vllm_argv(tmp_path)
    assert argv[argv.index("--max-model-len") + 1] == "4096"
    assert argv[-4:] == LISTENS


def test_the_pass_through_reaches_vllm_whole(tmp_path: Path, bin_dir: Path) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)
    override = '{"architectures": ["XLMRobertaForSequenceClassification"]}'

    result = run_app(
        tmp_path,
        bin_dir,
        "reranker",
        QP_VLLM_ARGS=f"--hf-overrides '{override}' --dtype bfloat16",
    )

    assert result.returncode == 0, result.stderr
    assert vllm_argv(tmp_path)[-8:-4] == [
        "--hf-overrides",
        override,
        "--dtype",
        "bfloat16",
    ]


def test_a_named_vllm_variable_is_exported_to_vllm(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(
        tmp_path,
        bin_dir,
        QP_VLLM_CPU_OMP_THREADS_BIND="0-7",
        QP_VLLM_CPU_RESERVED_CPUS="2",
    )

    assert result.returncode == 0, result.stderr
    started = started_env(tmp_path, "vllm")
    assert "VLLM_CPU_OMP_THREADS_BIND=0-7" in started
    assert "VLLM_CPU_NUM_OF_RESERVED_CPU=2" in started


def test_a_vllm_variable_given_both_ways_stops_the_app_before_vllm(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(
        tmp_path,
        bin_dir,
        QP_VLLM_CPU_OMP_THREADS_BIND="0-7",
        VLLM_CPU_OMP_THREADS_BIND="auto",
    )

    assert result.returncode == 1
    assert last_message(result).startswith("qp-vllm-config: ")
    assert "VLLM_CPU_OMP_THREADS_BIND" in last_message(result)
    assert vllm_argv(tmp_path) == []


def test_an_llm_without_a_source_id_stops_the_app_in_one_line(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(tmp_path, bin_dir, "llm")

    assert result.returncode == 1
    assert result.stderr.count("\n") == 1
    assert "components.llm.source_id is empty" in result.stderr
    assert vllm_argv(tmp_path) == []


def test_dominos_path_prefix_becomes_root_path(tmp_path: Path, bin_dir: Path) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(tmp_path, bin_dir, DOMINO_RUN_HOST_PATH="/apps/abc123/")

    assert result.returncode == 0, result.stderr
    argv = vllm_argv(tmp_path)
    assert argv[argv.index("--port") + 2 :] == ["--root-path", "/apps/abc123"]


def test_vllm_runs_in_the_use_cases_directory(tmp_path: Path, bin_dir: Path) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)
    use_case = tmp_path / "use_case"
    use_case.mkdir()

    result = run_app(tmp_path, bin_dir, QP_USE_CASE_DIR=str(use_case))

    assert result.returncode == 0, result.stderr
    assert started_in(tmp_path, "vllm") == use_case


def test_without_vllm_the_app_says_how_to_get_it(tmp_path: Path, bin_dir: Path) -> None:
    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 1
    message = last_message(result)
    assert message.startswith("app.sh[model-server vllm embedder]: ")
    for needed in ("'vllm'", "glibc 2.39", "Ubuntu 24.04", "QP_BOOTSTRAP=1", "uv"):
        assert needed in message
    assert log(tmp_path, "qp-vllm-config") == ""


def test_the_bootstrap_installs_nothing_when_vllm_is_there(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_vllm_stand_in(bin_dir / "vllm", tmp_path)

    result = run_app(tmp_path, bin_dir, QP_BOOTSTRAP="1")

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "uv") == ""
    assert vllm_argv(tmp_path)[0] == "serve"


def test_the_bootstrap_installs_the_vllm_group_alone_in_a_virtualenv_of_its_own(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(tmp_path, bin_dir, "reranker", QP_BOOTSTRAP="1")

    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "vllm-venv"
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",
        f"pip install --python {venv}/bin/python --group vllm --torch-backend cpu",
    ]
    # --group reads the toolkit's pyproject.toml, in the directory uv runs in.
    assert (tmp_path / "uv.cwd").read_text().splitlines()[1] == str(TOOLKIT_ROOT)
    assert vllm_argv(tmp_path, "venv-vllm")[:2] == ["serve", SERVED["reranker"][0]]
    # No libiomp5.so in the virtualenv: nothing is preloaded.
    assert (tmp_path / "venv-vllm.preload").read_text() == "\n"


def test_the_bootstrap_preloads_the_intel_openmp_vllm_brings(
    tmp_path: Path, bin_dir: Path
) -> None:
    library = tmp_path / "home" / "vllm-venv" / "lib" / "libiomp5.so"
    library.parent.mkdir(parents=True)
    library.write_text("")

    result = run_app(tmp_path, bin_dir, QP_BOOTSTRAP="1")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "venv-vllm.preload").read_text().strip() == str(library)


def test_a_failed_install_names_the_glibc_found(tmp_path: Path, bin_dir: Path) -> None:
    write_fake(
        bin_dir / "uv",
        tmp_path / "uv.log",
        'if [ "$1" = venv ]; then mkdir -p "$2/bin"; fi\n'
        'if [ "$1" = pip ]; then exit 1; fi\n',
    )
    write_fake(bin_dir / "ldd", tmp_path / "ldd.log", f"echo '{GLIBC_FOUND}'\n")

    result = run_app(tmp_path, bin_dir, "reranker", QP_BOOTSTRAP="1")

    assert result.returncode == 1
    message = last_message(result)
    assert message.startswith(
        "app.sh[model-server vllm reranker]: installing vLLM failed"
    )
    for needed in (
        "glibc 2.39",
        "Ubuntu 24.04",
        f"this environment has {GLIBC_FOUND};",
    ):
        assert needed in message
    assert vllm_argv(tmp_path, "venv-vllm") == []


def test_without_vllm_and_without_uv_the_app_stops_naming_ubuntu_24_04(
    tmp_path: Path,
) -> None:
    bare = bare_path(tmp_path)

    result = run_app(tmp_path, bare, QP_BOOTSTRAP="1", PATH=str(bare))

    assert result.returncode == 1
    message = last_message(result)
    for needed in ("'vllm'", "'uv'", "glibc 2.39", "Ubuntu 24.04"):
        assert needed in message


def test_a_vllm_app_installs_the_project_when_qp_vllm_config_is_missing(
    tmp_path: Path,
) -> None:
    bare = bare_path(tmp_path, "mkdir", "cp", "chmod")
    shutil.copy(make_stand_ins(tmp_path) / "uv", bare / "uv")
    write_vllm_stand_in(bare / "vllm", tmp_path)

    wheels = wheels_env(tmp_path)

    result = run_app(tmp_path, bare, PATH=str(bare), **wheels)

    assert result.returncode == 0, result.stderr
    venv, found = tmp_path / "home" / "qp-venv", wheels["QP_BOOTSTRAP_WHEELS"]
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",
        f"pip install --python {venv}/bin/python --find-links {found} "
        f"{CORE_ROOT} {TOOLKIT_ROOT}",
    ]
    assert log(tmp_path, "qp-vllm-config").splitlines() == [
        "--environment embedder",
        "--serve embedder",
    ]
    assert vllm_argv(tmp_path)[1] == SERVED["embedder"][0]


def test_without_qp_vllm_config_or_uv_the_app_names_both(tmp_path: Path) -> None:
    bare = bare_path(tmp_path)
    write_vllm_stand_in(bare / "vllm", tmp_path)

    result = run_app(tmp_path, bare, PATH=str(bare))

    assert result.returncode == 1
    message = last_message(result)
    assert "'qp-vllm-config'" in message
    assert "'uv'" in message
    assert vllm_argv(tmp_path) == []


def test_the_vllm_group_the_script_installs_is_pinned_in_pyproject() -> None:
    groups = re.findall(r"uv pip install .*--group (\S+)", VLLM_SH.read_text())
    pyproject = tomllib.loads((TOOLKIT_ROOT / "pyproject.toml").read_text())

    assert groups == ["vllm"]
    (pin,) = pyproject["dependency-groups"]["vllm"]
    assert pin.startswith("vllm @ https://github.com/vllm-project/vllm/releases/")
    assert "+cpu-" in pin
