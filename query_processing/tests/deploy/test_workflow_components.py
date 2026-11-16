"""The components of the workflow file, as qp-deploy reads them to resolve an alias."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.workflow_components import (
    WorkflowComponent,
    component_variable,
    read_workflow_components,
    refuse_override_values,
    refuse_unknown_override_keys,
    workflow_file_for,
    workflow_setting_path,
    workflow_variable,
)
from rag_graphs import FIXTURE_WORKFLOW


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


def test_the_extensions_the_file_names_are_loaded_before_its_components(
    tmp_path: Path,
) -> None:
    path = written(
        tmp_path,
        "extensions: [shout_extension]\ncomponents:\n  shouter: {backend: local}\n",
    )

    assert read_workflow_components(path) == {
        "shouter": WorkflowComponent(type="shouter", backend="local")
    }
    assert "shout_extension.shouting" in sys.modules


def test_an_override_of_an_extensions_component_is_checked_by_its_block(
    tmp_path: Path,
) -> None:
    path = written(
        tmp_path,
        (FIXTURE_WORKFLOW.read_text() + "\nextensions: [shout_extension]\n").replace(
            "components:\n", "components:\n  shouter: {backend: local}\n", 1
        ),
    )
    overrides = {"components": {"shouter": {"local": {"sufix": "!"}}}}

    refuse_unknown_override_keys(overrides, path)  # the type's keys are its own
    with pytest.raises(DominoDeployError, match=r"sufix.*expected.*suffix"):
        refuse_override_values(overrides, path)


def test_an_extension_that_cannot_be_imported_stops_the_deploy_tool(
    tmp_path: Path,
) -> None:
    path = written(tmp_path, "extensions: [no_such_package]\ncomponents: {}\n")

    with pytest.raises(
        DominoDeployError, match="extension 'no_such_package' cannot be imported"
    ):
        read_workflow_components(path)


def test_only_the_file_is_read_not_the_variables_of_this_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Apps do not see the Workspace's shell: the file says what they get.
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__BACKEND", "fake")
    assert read_workflow_components(FIXTURE_WORKFLOW)["reranker"].backend == "local"


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(DominoDeployError) as caught:
        read_workflow_components(tmp_path / "absent.yaml")
    assert str(caught.value) == (
        "qp-deploy reads the components of the workflow file: workflow file "
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
    assert workflow_file_for({"QP_WORKFLOW_FILE": "lexical.yaml"}) == Path(
        "lexical.yaml"
    )


def test_without_one_the_file_is_found_as_qp_serve_finds_it() -> None:
    # tests/conftest.py names the tests' file in QP_WORKFLOW_FILE.
    assert workflow_file_for({"QP_ES_URL": "https://es"}) == FIXTURE_WORKFLOW


def test_a_components_value_is_set_by_the_variable_its_path_names() -> None:
    assert component_variable("reranker_fr", "served", "url") == (
        "QP_COMPONENTS__RERANKER_FR__SERVED__URL"
    )
    assert component_variable("embedder", "backend") == (
        "QP_COMPONENTS__EMBEDDER__BACKEND"
    )


def test_a_value_of_the_workflow_file_is_set_by_the_variable_of_its_path() -> None:
    assert workflow_variable(("http", "retries")) == "QP_HTTP__RETRIES"
    assert workflow_variable(("run_label",)) == "QP_RUN_LABEL"


@pytest.mark.parametrize(
    ("variable", "path"),
    [
        ("QP_HTTP__TIMEOUT_S", ("http", "timeout_s")),
        ("QP_COMPONENTS__VECTOR_STORE__URL", ("components", "vector_store", "url")),
        ("qp_graph__stages__rerank__top_k", ("graph", "stages", "rerank", "top_k")),
        ("QP_RUN_LABEL", ("run_label",)),
        ("QP_TOPOLOGY_ALIAS", ("topology_alias",)),
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
        "QP_OVMS_MODELS_DIR",  # read by a launch script
        "QP_BOOTSTRAP_EXTRAS",
        "QP_WORKFLOW_FILE",  # which file, not a value of it
        "QP_DEPLOY_TOPOLOGY",  # the deploy tool's own
        "QP_DOMINO_API_PROXY",  # Domino's own run variable is read without QP_
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
        "  stages:\n"
        "    rerank: {call: rr, top_k: 5, reads: [standalone_query, candidates],\n"
        "             writes: [reranked]}\n"
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
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__API__URL", "https://r/v2/rerank")
    with pytest.raises(DominoDeployError, match="QP_COMPONENTS__RERANKER__API__URL"):
        refuse_override_values(
            {"components": {"reranker": {"backend": "api"}}}, FIXTURE_WORKFLOW
        )


def prefilter_override(condition: dict[str, object]) -> dict[str, object]:
    """The overrides that give the fixture's vector store a one-condition prefilter."""
    return {"components": {"vector_store": {"prefilter": [condition]}}}


def test_a_prefilter_override_on_undeclared_fields_passes() -> None:
    overrides = prefilter_override(
        {"attribute": "active", "comparator": "eq", "value": True}
    )

    refuse_unknown_override_keys(overrides, FIXTURE_WORKFLOW)
    refuse_override_values(overrides, FIXTURE_WORKFLOW)


@pytest.mark.parametrize("block", ["api", "served"])
def test_an_override_of_the_llms_structured_output_passes(block: str) -> None:
    overrides = {
        "components": {"llm": {block: {"structured_output": "format_instructions"}}}
    }

    refuse_unknown_override_keys(overrides, FIXTURE_WORKFLOW)
    refuse_override_values(overrides, FIXTURE_WORKFLOW)


def test_a_prefilter_override_of_the_wrong_shape_is_refused() -> None:
    overrides = prefilter_override(
        {"attribute": "version", "comparator": "in", "value": []}
    )

    with pytest.raises(DominoDeployError, match="'in' takes a list of at least one"):
        refuse_override_values(overrides, FIXTURE_WORKFLOW)
