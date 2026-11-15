"""Where the workflow file is found, and the settings source that reads it."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from query_processing.config.workflow_file import (
    WORKFLOW_FILE_VARIABLE,
    WorkflowFileNotFoundError,
    declared_alias,
    unknown_keys,
    workflow_file,
    workflow_file_source,
)
from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import FIXTURE_WORKFLOW
from repo_paths import TOOLKIT_ROOT


@pytest.fixture
def elsewhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty working directory, with ``QP_WORKFLOW_FILE`` unset."""
    monkeypatch.delenv(WORKFLOW_FILE_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_with_no_file_here_and_none_named_it_stops_saying_where_it_looked(
    elsewhere: Path,
) -> None:
    with pytest.raises(WorkflowFileNotFoundError) as caught:
        workflow_file()

    assert str(caught.value) == (
        "no workflow file: QP_WORKFLOW_FILE is not set, and the working directory "
        f"{elsewhere} has no workflow.yaml; run the command from a use case's "
        "directory, or set QP_WORKFLOW_FILE to a workflow file"
    )
    # A settings mistake to every command: each turns a ValueError into a
    # usage error, and an OSError where it reads files.
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, FileNotFoundError)


def test_the_toolkit_ships_no_workflow_file() -> None:
    # A use case holds its own: none ships with the code, as data or as a default.
    assert sorted((TOOLKIT_ROOT / "src").rglob("*.yaml")) == []
    pyproject = tomllib.loads((TOOLKIT_ROOT / "pyproject.toml").read_text("utf-8"))
    data = pyproject["tool"]["setuptools"]["package-data"]
    shipped = [name for files in data.values() for name in files]
    assert shipped, "no package data read: the lookup is wrong"
    assert [name for name in shipped if name.endswith((".yaml", ".yml"))] == []


def test_with_no_file_the_settings_stop_saying_where_they_looked(
    elsewhere: Path,
) -> None:
    with pytest.raises(WorkflowFileNotFoundError, match="no workflow file: "):
        WorkflowSettings(_env_file=None)  # type: ignore[call-arg]


def test_the_tests_read_their_own_fixture_file() -> None:
    assert workflow_file() == FIXTURE_WORKFLOW


def test_a_workflow_yaml_in_the_working_directory_is_used(elsewhere: Path) -> None:
    (elsewhere / "workflow.yaml").write_text("graph: {}\n", encoding="utf-8")

    assert workflow_file().resolve() == (elsewhere / "workflow.yaml").resolve()


def test_the_file_a_variable_names_comes_first(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (elsewhere / "workflow.yaml").write_text("graph: {}\n", encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(elsewhere / "variant.yaml"))

    assert workflow_file() == elsewhere / "variant.yaml"


def test_the_source_reads_the_files_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text("http: {retries: 0}\ngraph: {stages: {}}\n", encoding="utf-8")

    assert workflow_file_source(WorkflowSettings, path)() == {
        "http": {"retries": 0},
        "graph": {"stages": {}},
    }


def test_the_source_names_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "workflow.yaml"

    with pytest.raises(FileNotFoundError, match=r"nowhere/workflow\.yaml"):
        workflow_file_source(WorkflowSettings, missing)


def test_the_source_refuses_a_key_that_is_no_setting_naming_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text("graph: {stages: {}}\nretrieval_mode: lexical\n", encoding="utf-8")

    with pytest.raises(ValueError, match="'retrieval_mode'") as refused:
        workflow_file_source(WorkflowSettings, path)

    assert "workflow.yaml" in str(refused.value)


def test_unknown_keys_are_those_no_field_is_named_after() -> None:
    keys = ["graph", "zeta", "es_index", "local_models", "alpha"]

    assert unknown_keys(keys, WorkflowSettings) == ["alpha", "es_index", "zeta"]


def test_a_declared_alias_is_read_from_the_file_alone(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        "components:\n"
        "  reranker: {backend: local, source_id: /models/rr/}\n"
        "  embedder: {backend: local, source_id: org/e, alias: emb}\n"
        "  llm: {backend: fake}\n",
        encoding="utf-8",
    )

    assert declared_alias("reranker", path) == "rr"
    assert declared_alias("embedder", path) == "emb"
    assert declared_alias("llm", path) == ""  # no source id
    assert declared_alias("absent", path) == ""
    assert declared_alias("reranker") == "fixture-reranker"


def test_a_declared_alias_needs_a_workflow_file(elsewhere: Path) -> None:
    with pytest.raises(WorkflowFileNotFoundError, match="no workflow file: "):
        declared_alias("reranker")
