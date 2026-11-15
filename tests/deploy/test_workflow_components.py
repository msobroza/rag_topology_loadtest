"""The components of the workflow file, as rag-deploy reads them to resolve an alias."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_load_test.config.workflow_file import PACKAGED_WORKFLOW
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.workflow_components import (
    WorkflowComponent,
    component_variable,
    read_workflow_components,
    workflow_file_for,
)


def written(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(text)
    return path


def test_the_packaged_file_declares_the_rags_four_components_in_order() -> None:
    assert read_workflow_components(PACKAGED_WORKFLOW) == {
        "embedder": WorkflowComponent(type="embedder", backend="local"),
        "reranker": WorkflowComponent(type="reranker", backend="local"),
        "llm": WorkflowComponent(type="llm", backend="api"),
        "vector_store": WorkflowComponent(type="vector_store", backend="elasticsearch"),
    }


def test_a_type_defaults_to_the_name_unless_the_file_states_it(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        "components:\n"
        "  reranker: {backend: none}\n"
        "  reranker_fr: {type: reranker, backend: served}\n",
    )
    assert read_workflow_components(path) == {
        "reranker": WorkflowComponent(type="reranker", backend="none"),
        "reranker_fr": WorkflowComponent(type="reranker", backend="served"),
    }


def test_only_the_file_is_read_not_the_variables_of_this_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Apps do not see the Workspace's shell: the file says what they get.
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__BACKEND", "fake")
    assert read_workflow_components(PACKAGED_WORKFLOW)["reranker"].backend == "local"


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(DominoDeployError) as caught:
        read_workflow_components(tmp_path / "absent.yaml")
    assert str(caught.value) == (
        "rag-deploy reads the components of the workflow file: workflow file "
        f"'{tmp_path / 'absent.yaml'}' does not exist"
    )


def test_an_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    path = written(tmp_path, "componets: {}\n")
    with pytest.raises(DominoDeployError, match="unknown key\\(s\\) 'componets'"):
        read_workflow_components(path)


@pytest.mark.parametrize(
    ("text", "what", "value"),
    [
        ("components: [embedder]\n", "its components block", "['embedder']"),
        ("components: {embedder: local}\n", "component 'embedder'", "'local'"),
    ],
)
def test_a_components_block_of_another_shape_is_refused(
    tmp_path: Path, text: str, what: str, value: str
) -> None:
    path = written(tmp_path, text)
    with pytest.raises(DominoDeployError) as caught:
        read_workflow_components(path)
    assert str(caught.value) == (
        f"workflow file '{path}': {what} is {value}; expected a mapping, such as "
        "{embedder: {backend: local, source_id: ...}}"
    )


def test_the_workflow_apps_own_file_is_the_one_read() -> None:
    assert workflow_file_for({"RAG_WORKFLOW_FILE": "lexical.yaml"}) == Path(
        "lexical.yaml"
    )


def test_without_one_the_file_is_found_as_rag_serve_finds_it() -> None:
    # tests/conftest.py names the packaged file in RAG_WORKFLOW_FILE.
    assert workflow_file_for({"RAG_ES_URL": "https://es"}) == PACKAGED_WORKFLOW


def test_a_components_value_is_set_by_the_variable_its_path_names() -> None:
    assert component_variable("reranker_fr", "served", "url") == (
        "RAG_COMPONENTS__RERANKER_FR__SERVED__URL"
    )
    assert component_variable("embedder", "backend") == (
        "RAG_COMPONENTS__EMBEDDER__BACKEND"
    )
