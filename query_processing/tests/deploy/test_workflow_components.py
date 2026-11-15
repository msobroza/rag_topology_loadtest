"""The components of the workflow file, as rag-deploy reads them to resolve an alias."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_graphs import FIXTURE_WORKFLOW
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.workflow_components import (
    WorkflowComponent,
    component_variable,
    read_workflow_components,
    refuse_override_values,
    refuse_unknown_override_keys,
    workflow_file_for,
    workflow_setting_path,
    workflow_variable,
)


def written(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(text)
    return path


def test_the_fixture_file_declares_four_components_in_order() -> None:
    assert read_workflow_components(FIXTURE_WORKFLOW) == {
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
    assert read_workflow_components(FIXTURE_WORKFLOW)["reranker"].backend == "local"


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
    # tests/conftest.py names the tests' file in RAG_WORKFLOW_FILE.
    assert workflow_file_for({"RAG_ES_URL": "https://es"}) == FIXTURE_WORKFLOW


def test_a_components_value_is_set_by_the_variable_its_path_names() -> None:
    assert component_variable("reranker_fr", "served", "url") == (
        "RAG_COMPONENTS__RERANKER_FR__SERVED__URL"
    )
    assert component_variable("embedder", "backend") == (
        "RAG_COMPONENTS__EMBEDDER__BACKEND"
    )


def test_a_value_of_the_workflow_file_is_set_by_the_variable_of_its_path() -> None:
    assert workflow_variable(("http", "retries")) == "RAG_HTTP__RETRIES"
    assert workflow_variable(("run_label",)) == "RAG_RUN_LABEL"


@pytest.mark.parametrize(
    ("variable", "path"),
    [
        ("RAG_HTTP__TIMEOUT_S", ("http", "timeout_s")),
        ("RAG_COMPONENTS__VECTOR_STORE__URL", ("components", "vector_store", "url")),
        ("rag_graph__stages__rerank__top_k", ("graph", "stages", "rerank", "top_k")),
        ("RAG_RUN_LABEL", ("run_label",)),
        ("RAG_TOPOLOGY_ALIAS", ("topology_alias",)),
    ],
)
def test_a_variable_of_the_workflow_settings_names_its_path(
    variable: str, path: tuple[str, ...]
) -> None:
    assert workflow_setting_path(variable) == path
    assert workflow_setting_path(workflow_variable(path)) == path  # both ways


@pytest.mark.parametrize(
    "variable",
    [
        "RAG_OVMS_MODELS_DIR",  # read by a launch script
        "RAG_BOOTSTRAP_EXTRAS",
        "RAG_WORKFLOW_FILE",  # which file, not a value of it
        "RAG_DEPLOY_TOPOLOGY",  # the deploy tool's own
        "RAG_DOMINO_API_PROXY",  # Domino's own run variable is read without RAG_
        "DOMINO_API_PROXY",
        "OPENAI_API_KEY",
    ],
)
def test_a_variable_that_is_no_value_of_the_workflow_file_has_no_path(
    variable: str,
) -> None:
    assert workflow_setting_path(variable) is None


def test_overrides_of_declared_names_and_known_keys_pass(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        "components:\n"
        "  rr: {type: reranker, backend: local, source_id: org/r}\n"
        "graph:\n"
        "  stages: {rerank: {call: rr, top_k: 5}}\n"
        "  edges: [[START, rerank, END]]\n",
    )
    overrides = {
        "components": {"rr": {"local": {"threads": 2}, "http": {"retries": 0}}},
        "graph": {"stages": {"rerank": {"top_k": 3, "call": "rr"}}},
        "calibration": {"rounds": 2},
    }
    refuse_unknown_override_keys(overrides, path)  # rr is a reranker: it has local
    refuse_override_values(overrides, path)


def test_the_values_are_checked_with_the_overrides_over_the_file_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A variable of this shell would give the URL; the App does not see it.
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__API__URL", "https://r/v2/rerank")
    with pytest.raises(DominoDeployError, match="RAG_COMPONENTS__RERANKER__API__URL"):
        refuse_override_values(
            {"components": {"reranker": {"backend": "api"}}}, FIXTURE_WORKFLOW
        )
