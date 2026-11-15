"""Where the workflow file is found, and the settings source that reads it."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import (
    PACKAGED_WORKFLOW,
    WORKFLOW_FILE_VARIABLE,
    declared_alias,
    unknown_keys,
    workflow_file,
    workflow_file_source,
)


@pytest.fixture
def elsewhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty working directory, with ``RAG_WORKFLOW_FILE`` unset."""
    monkeypatch.delenv(WORKFLOW_FILE_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_with_no_file_here_and_none_named_the_packaged_file_is_used(
    elsewhere: Path,
) -> None:
    assert workflow_file() == PACKAGED_WORKFLOW
    assert PACKAGED_WORKFLOW.is_file()


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

    assert workflow_file_source(RagSettings, path)() == {
        "http": {"retries": 0},
        "graph": {"stages": {}},
    }


def test_the_source_names_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "workflow.yaml"

    with pytest.raises(FileNotFoundError, match=r"nowhere/workflow\.yaml"):
        workflow_file_source(RagSettings, missing)


def test_the_source_refuses_a_key_that_is_no_setting_naming_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text("graph: {stages: {}}\nretrieval_mode: lexical\n", encoding="utf-8")

    with pytest.raises(ValueError, match="'retrieval_mode'") as refused:
        workflow_file_source(RagSettings, path)

    assert "workflow.yaml" in str(refused.value)


def test_unknown_keys_are_those_no_field_is_named_after() -> None:
    keys = ["graph", "zeta", "es_index", "local_models", "alpha"]

    assert unknown_keys(keys, RagSettings) == ["alpha", "es_index", "zeta"]


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
    assert declared_alias("reranker") == "crossencoder-camembert-base-mmarcoFR"
