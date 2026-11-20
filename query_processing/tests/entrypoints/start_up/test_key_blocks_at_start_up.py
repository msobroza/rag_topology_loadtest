"""What ``qp-serve`` refuses of an API key's settings, a usage error, before it serves.

The old ``api_key`` of an endpoint or of the Elasticsearch store, set in the
environment, in ``.env`` or in a workflow file, is refused in one line that
names the component and the ``key`` block to write, and never its value. A ``key`` block whose source reads none of
a key it holds, or lacks one it reads, is refused naming its variable. An
unknown source is refused with the other unknown names
(``test_unknown_names_at_start_up.py``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES, shell_settings
from rag_graphs import FIXTURE_WORKFLOW

SECRET = "sk-never-printed-4711"


class KeyedBlock(NamedTuple):
    """A block that holds a ``key`` block: its component and its path under it."""

    component: str
    block: str

    @property
    def path(self) -> str:
        return f"components.{self.component}.{self.block}"

    @property
    def variable(self) -> str:
        return f"QP_COMPONENTS__{self.component}__{self.block}".upper()


# Each block that held an api_key before key sources, and holds a key block now.
KEYED_BLOCKS = pytest.mark.parametrize(
    "keyed",
    [KeyedBlock("llm", "api"), KeyedBlock("vector_store", "elasticsearch")],
    ids=["llm-api", "vector-store-elasticsearch"],
)


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


SetOldKey = Callable[[pytest.MonkeyPatch, Path, KeyedBlock], None]


def old_key_in_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keyed: KeyedBlock
) -> None:
    monkeypatch.setenv(f"{keyed.variable}__API_KEY", SECRET)


def old_key_in_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keyed: KeyedBlock
) -> None:
    line = f"{keyed.variable}__API_KEY={SECRET}\n"
    (tmp_path / ".env").write_text(line, encoding="utf-8")


def old_key_in_the_workflow_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keyed: KeyedBlock
) -> None:
    workflow = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    component = workflow["components"][keyed.component]
    component.setdefault(keyed.block, {})["api_key"] = SECRET
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(workflow), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))


@KEYED_BLOCKS
@pytest.mark.parametrize(
    "set_old_key",
    [old_key_in_the_environment, old_key_in_dotenv, old_key_in_the_workflow_file],
    ids=["environment", "dotenv", "workflow-file"],
)
def test_the_old_api_key_is_refused_naming_the_component_and_the_key_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    set_old_key: SetOldKey,
    keyed: KeyedBlock,
) -> None:
    set_old_key(monkeypatch, tmp_path, keyed)

    line = refusal_of_qp_serve(capsys, f"{keyed.variable}__API_KEY")

    assert f"{keyed.path}.api_key" in line
    assert f"the {keyed.component} component" in line
    assert f"{keyed.path}.key" in line
    assert SECRET not in line


@KEYED_BLOCKS
@pytest.mark.parametrize(
    ("block", "said"),
    [
        (
            {"SOURCE": "domino", "VARIABLE": "OPENAI_API_KEY"},
            "the 'domino' key source reads no key beside 'source'; got 'variable'",
        ),
        (
            {"SOURCE": "env"},
            "the 'env' key source reads 'variable' beside 'source'; got none",
        ),
        (
            {"SOURCE": "env", "VARIABLE": "OPENAI_API_KEY", "VALUE": "x"},
            "the 'env' key source reads 'variable' beside 'source'; got 'value', "
            "'variable'",
        ),
    ],
    ids=["variable-beside-domino", "env-without-variable", "a-key-env-does-not-read"],
)
def test_a_key_block_holding_what_its_source_does_not_read_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    block: dict[str, str],
    said: str,
    keyed: KeyedBlock,
) -> None:
    # Every component is on its fake backend: each key block is checked all the same.
    for key, value in block.items():
        monkeypatch.setenv(f"{keyed.variable}__KEY__{key}", value)

    line = refusal_of_qp_serve(capsys, f"{keyed.variable}__KEY")

    assert f"{keyed.variable}__KEY: {said}" in line
