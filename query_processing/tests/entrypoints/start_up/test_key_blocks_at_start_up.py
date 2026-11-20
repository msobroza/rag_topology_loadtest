"""What ``qp-serve`` refuses of an endpoint's key, as a usage error, before it serves.

The old ``api_key`` of an endpoint, set in the environment, in ``.env`` or in a
workflow file, is refused in one line that names the component and the ``key``
block to write, and never its value. A ``key`` block whose source reads none of
a key it holds, or lacks one it reads, is refused naming its variable. An
unknown source is refused with the other unknown names
(``test_unknown_names_at_start_up.py``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES, shell_settings
from rag_graphs import FIXTURE_WORKFLOW

SECRET = "sk-never-printed-4711"
OLD_VARIABLE = "QP_COMPONENTS__LLM__API__API_KEY"


@pytest.fixture(autouse=True)
def empty_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All-fake backends, no QP_* from the shell, an empty directory, no server."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qp_serve.uvicorn, "run", never_serve)


def never_serve(app: object, **options: object) -> None:
    raise AssertionError(f"a server started with {options}: nothing was refused")


def refusal_of_qp_serve(capsys: pytest.CaptureFixture[str], naming: str) -> str:
    """Run ``qp-serve``; return the one line of its refusal that names ``naming``."""
    with pytest.raises(SystemExit) as exited:
        qp_serve.main([])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: qp-serve")
    assert "Traceback" not in err
    assert SECRET not in err
    [line] = [line for line in err.splitlines() if naming in line]
    return line


def old_key_in_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(OLD_VARIABLE, SECRET)


def old_key_in_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{OLD_VARIABLE}={SECRET}\n", encoding="utf-8")


def old_key_in_the_workflow_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workflow = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    workflow["components"]["llm"]["api"]["api_key"] = SECRET
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(workflow), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))


@pytest.mark.parametrize(
    "set_old_key",
    [old_key_in_the_environment, old_key_in_dotenv, old_key_in_the_workflow_file],
    ids=["environment", "dotenv", "workflow-file"],
)
def test_the_old_api_key_is_refused_naming_the_component_and_the_key_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    set_old_key: Callable[[pytest.MonkeyPatch, Path], None],
) -> None:
    set_old_key(monkeypatch, tmp_path)

    line = refusal_of_qp_serve(capsys, OLD_VARIABLE)

    assert "components.llm.api.api_key" in line
    assert "the llm component" in line
    assert "components.llm.api.key" in line
    assert SECRET not in line


@pytest.mark.parametrize(
    ("block", "said"),
    [
        (
            {"SOURCE": "domino", "VARIABLE": "OPENAI_API_KEY"},
            "QP_COMPONENTS__LLM__API__KEY: the 'domino' key source reads no key "
            "beside 'source'; got 'variable'",
        ),
        (
            {"SOURCE": "env"},
            "QP_COMPONENTS__LLM__API__KEY: the 'env' key source reads 'variable' "
            "beside 'source'; got none",
        ),
        (
            {"SOURCE": "env", "VARIABLE": "OPENAI_API_KEY", "VALUE": "x"},
            "QP_COMPONENTS__LLM__API__KEY: the 'env' key source reads 'variable' "
            "beside 'source'; got 'value', 'variable'",
        ),
    ],
    ids=["variable-beside-domino", "env-without-variable", "a-key-env-does-not-read"],
)
def test_a_key_block_holding_what_its_source_does_not_read_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    block: dict[str, str],
    said: str,
) -> None:
    # The LLM is on its fake backend: every key block is checked all the same.
    for key, value in block.items():
        monkeypatch.setenv(f"QP_COMPONENTS__LLM__API__KEY__{key}", value)

    line = refusal_of_qp_serve(capsys, "QP_COMPONENTS__LLM__API__KEY")

    assert said in line
