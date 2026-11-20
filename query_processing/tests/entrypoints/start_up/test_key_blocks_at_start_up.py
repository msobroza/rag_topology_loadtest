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


# --- one place checks a key block before the settings library sees it ----------------
# The library quotes the value of what it refuses: a key typed where a block's
# source or variable goes, or where a block goes, would be printed.
LLM_KEY = "QP_COMPONENTS__LLM__API__KEY"


def key_block_in_the_workflow_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, block: object
) -> None:
    workflow = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    workflow["components"]["llm"].setdefault("api", {})["key"] = block
    path = tmp_path / "keyed.yaml"
    path.write_text(yaml.safe_dump(workflow), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))


@pytest.mark.parametrize(
    ("variables", "said"),
    [
        ({f"{LLM_KEY}__VALUE": SECRET}, f"{LLM_KEY}__SOURCE is unset"),
        ({LLM_KEY: '{"value": "' + SECRET + '"}'}, f"{LLM_KEY}__SOURCE is unset"),
        ({LLM_KEY: SECRET}, f"{LLM_KEY} (components.llm.api.key) is not a key block"),
        (
            {f"{LLM_KEY}__SOURCE": SECRET},
            f"{LLM_KEY}__SOURCE (components.llm.api.key.source) is not a name",
        ),
        (
            {f"{LLM_KEY}__SOURCE": "env", f"{LLM_KEY}__VARIABLE": SECRET},
            f"{LLM_KEY}__VARIABLE (components.llm.api.key.variable) is not a name",
        ),
    ],
    ids=["no-source", "no-source-as-json", "a-key-for-a-block", "source", "variable"],
)
def test_a_key_typed_in_a_key_block_by_a_variable_is_refused_and_never_printed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variables: dict[str, str],
    said: str,
) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    line = refusal_of_qp_serve(capsys, said.split(" ")[0])

    assert said in line


@pytest.mark.parametrize(
    ("value", "kind"),
    [(123, "int"), ({"name": SECRET}, "dict"), ([SECRET], "list"), (None, "NoneType")],
    ids=["number", "mapping", "list", "null"],
)
def test_a_value_beside_source_that_is_not_text_is_refused_and_never_printed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    value: object,
    kind: str,
) -> None:
    block = {"source": "env", "variable": value}
    key_block_in_the_workflow_file(monkeypatch, tmp_path, block)

    line = refusal_of_qp_serve(capsys, f"{LLM_KEY}__VARIABLE")

    assert f"(components.llm.api.key.variable) is a {kind}: expected text" in line


@pytest.mark.parametrize(
    ("variable", "place"),
    [
        ("QP_COMPONENTS__RERANKER__SERVED__API_KEY", "components.reranker.served"),
        (
            "QP_COMPONENTS__VECTOR_STORE__CHROMA__API_KEY",
            "components.vector_store.chroma",
        ),
        ("QP_COMPONENTS__LLM__API_KEY", "components.llm"),
    ],
    ids=["served", "chroma", "the-component-itself"],
)
def test_an_old_api_key_under_any_block_of_a_component_gets_the_old_keys_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variable: str,
    place: str,
) -> None:
    monkeypatch.setenv(variable, SECRET)

    line = refusal_of_qp_serve(capsys, variable)

    assert f"{variable} ({place}.api_key in a workflow file) is no longer read" in line


def test_the_old_keys_line_names_the_variable_the_key_block_already_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    block = {"source": "env", "variable": "OPENAI_API_KEY"}
    key_block_in_the_workflow_file(monkeypatch, tmp_path, block)
    monkeypatch.setenv("QP_COMPONENTS__LLM__API__API_KEY", SECRET)

    line = refusal_of_qp_serve(capsys, "QP_COMPONENTS__LLM__API__API_KEY")

    assert "components.llm.api.key" in line
    assert "set OPENAI_API_KEY to the API key" in line
