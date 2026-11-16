"""The workflow app runs the graph of its workflow file, seen through its HTTP API.

The app is started from all-fake settings, its reranker a recording double that
keeps retrieval order, so a test reads the sizes each stage was given. The app
runs the file ``QP_WORKFLOW_FILE`` names, else ``workflow.yaml`` in the working
directory, the use case's; with neither it does not start, saying where it
looked (test_no_workflow_file_at_start_up.py): no workflow file ships with the
code. A variable named by a value's path overrides that value.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from query_processing.application.graph_check import check_graph
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints import qp_serve
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS
from rag_graphs import fixture_graph, write_workflow_file
from workflow_fakes import RecordingEmbedder, RecordingReranker, ScriptedStore

QUESTION = "How many paid vacation days do employees accrue each year?"
TIMING_KEYS = {"embed", "retrieve", "rerank", "generate", "total"}
# The graph of the README's lexical workflow file, its prompts written out: no
# embed stage.
LEXICAL_WORKFLOW = """
graph:
  stages:
    retrieve:
      call: vector_store
      retrieval_mode: lexical
      top_k: 20
      reads: [standalone_query, metadata_filter]
      writes: [candidates]
    rerank:
      call: reranker
      top_k: 10
      reads: [standalone_query, candidates]
      writes: [reranked]
    generate:
      call: llm
      system_prompt: Answer from the numbered passages only.
      user_prompt: "Context: {context} Question: {question}"
      reads: [conversational_query, reranked]
      writes: [answer]
  edges:
    - [START, retrieve, rerank]
    - [generate, END]
  conditional_edges:
    - {source: rerank, path: is_query_mode, reads: [mode], path_map: {true: generate, false: END}}
"""
SettingsFactory = Callable[..., WorkflowSettings]


def ask(
    settings: WorkflowSettings, path: str = "/retrieve", **ports: Any
) -> dict[str, Any]:
    """Start the app on ``settings`` and ``ports``, ask ``path`` once, return the body."""
    with TestClient(create_app(CompositionRoot(settings, **ports))) as app:
        response = app.post(path, json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response.json()


def stage_changed(stage: str, **settings: object) -> dict[str, Any]:
    """The fixture's graph with one stage's settings changed."""
    graph = fixture_graph()
    graph["stages"][stage].update(settings)
    return graph


@pytest.fixture
def no_named_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run in an empty directory, with no ``QP_WORKFLOW_FILE``; return the directory."""
    monkeypatch.delenv(WORKFLOW_FILE_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- where the file comes from ----------------------------------------------------


def test_the_app_runs_the_graph_of_the_file_it_is_given(
    settings_factory: SettingsFactory,
) -> None:
    embedder, reranker = RecordingEmbedder(), RecordingReranker()
    settings = settings_factory(**FAKE_BACKENDS)  # the tests' file (conftest)

    body = ask(settings, "/query", embedder=embedder, reranker=reranker)

    assert set(body["timings_ms"]) == TIMING_KEYS
    assert embedder.queries == [QUESTION]  # embed, then a dense retrieval
    assert reranker.calls == [(QUESTION, 20, 10)]
    assert len(body["passages"]) == 10
    # FakeLLM echoes the user message: the file's user prompt, filled.
    assert body["answer"].startswith("[fake-llm] Context:\n[1] ")


def test_a_workflow_file_in_the_working_directory_is_used_in_its_place(
    settings_factory: SettingsFactory, no_named_file: Path
) -> None:
    write_workflow_file(
        no_named_file / "workflow.yaml", stage_changed("rerank", top_k=3)
    )
    reranker = RecordingReranker()

    body = ask(settings_factory(**FAKE_BACKENDS), reranker=reranker)

    assert reranker.calls == [(QUESTION, 20, 3)]
    assert len(body["passages"]) == 3


def test_the_file_a_variable_names_is_used_before_the_working_directorys(
    settings_factory: SettingsFactory,
    no_named_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_workflow_file(
        no_named_file / "workflow.yaml", stage_changed("rerank", top_k=3)
    )
    named = write_workflow_file(
        no_named_file / "variant.yaml", stage_changed("rerank", top_k=4)
    )
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(named))
    reranker = RecordingReranker()

    ask(settings_factory(**FAKE_BACKENDS), reranker=reranker)

    assert reranker.calls == [(QUESTION, 20, 4)]


def test_a_variable_naming_a_missing_file_fails_naming_it(
    settings_factory: SettingsFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(tmp_path / "nowhere.yaml"))

    with pytest.raises(FileNotFoundError, match=r"nowhere\.yaml"):
        settings_factory(**FAKE_BACKENDS)


# --- stages ------------------------------------------------------------------------


def test_a_stage_named_otherwise_is_an_instance_of_the_type_it_names(
    settings_factory: SettingsFactory, no_named_file: Path
) -> None:
    graph = fixture_graph()
    graph["stages"]["search"] = {**graph["stages"].pop("retrieve"), "type": "retrieve"}
    graph["stages"]["search"]["top_k"] = 4
    graph["edges"][0] = ["START", "embed", "search", "rerank"]
    write_workflow_file(no_named_file / "workflow.yaml", graph)
    reranker = RecordingReranker()

    body = ask(settings_factory(**FAKE_BACKENDS), reranker=reranker)

    assert reranker.calls == [(QUESTION, 4, 10)]
    assert len(body["passages"]) == 4


def test_stages_of_no_stage_type_that_state_their_lists_answer_as_plain_calls(
    settings_factory: SettingsFactory, no_named_file: Path
) -> None:
    graph = fixture_graph()
    del graph["stages"]["embed"], graph["stages"]["rerank"]
    graph["stages"]["embed_query"] = {
        "call": "embedder",
        "reads": ["standalone_query"],
        "writes": ["query_embedding"],
    }
    graph["stages"]["keep_best"] = {
        "call": "reranker",
        "top_k": 4,
        "reads": ["standalone_query", "candidates"],
        "writes": ["reranked"],
    }
    graph["edges"][0] = ["START", "embed_query", "retrieve", "keep_best"]
    graph["conditional_edges"][0]["source"] = "keep_best"
    write_workflow_file(no_named_file / "workflow.yaml", graph)
    settings = settings_factory(**FAKE_BACKENDS)
    embedder, reranker = RecordingEmbedder(), RecordingReranker()

    check_graph(settings.graph, settings.component_types())
    body = ask(settings, "/query", embedder=embedder, reranker=reranker)

    assert embedder.queries == [QUESTION]
    assert reranker.calls == [(QUESTION, 20, 4)]
    assert len(body["passages"]) == 4
    assert body["answer"].startswith("[fake-llm] Context:\n[1] ")


def test_a_stage_that_names_no_component_is_refused_at_start_up(
    settings_factory: SettingsFactory, no_named_file: Path
) -> None:
    graph = fixture_graph()
    del graph["stages"]["rerank"]["call"]
    write_workflow_file(no_named_file / "workflow.yaml", graph)

    with pytest.raises(ValidationError, match=r"graph\.stages\.rerank\.call"):
        settings_factory(**FAKE_BACKENDS)


# --- the environment wins over the file --------------------------------------------


def test_a_variable_named_by_a_values_path_overrides_the_file(
    settings_factory: SettingsFactory,
    no_named_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_workflow_file(
        no_named_file / "workflow.yaml", stage_changed("rerank", top_k=5)
    )
    monkeypatch.setenv("QP_GRAPH__STAGES__RERANK__TOP_K", "3")
    reranker = RecordingReranker()

    body = ask(settings_factory(**FAKE_BACKENDS), reranker=reranker)

    assert reranker.calls == [(QUESTION, 20, 3)]  # the file's other values stay
    assert len(body["passages"]) == 3


def test_a_variable_overrides_a_prompt_of_the_generate_stage(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    variable = "QP_GRAPH__STAGES__GENERATE__USER_PROMPT"
    monkeypatch.setenv(variable, "ASK {question} USING {context}")

    body = ask(settings_factory(**FAKE_BACKENDS), "/query")

    assert body["answer"].startswith(f"[fake-llm] ASK {QUESTION} USING [1] ")


# --- refusals at start-up ------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value", "refused"),
    [
        ("QP_GRAPH__STAGES__RERANK__TOP_N", "3", "top_n"),
        ("QP_GRAPH__STAGES__GENERATE__USER_PROMPT", "Q: {question}", "{context}"),
    ],
)
def test_a_stage_setting_its_stage_type_refuses_stops_the_app_at_start_up(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    refused: str,
) -> None:
    monkeypatch.setenv(variable, value)
    app = create_app(CompositionRoot(settings_factory(**FAKE_BACKENDS)))

    with pytest.raises(ValueError, match=refused), TestClient(app):
        pass  # the lifespan builds the graph before the first request


@pytest.mark.parametrize(
    ("workflow", "refused"),
    [
        ("grpah: {}\n", "unknown key(s) 'grpah'"),
        ("graph: {stages: {}, edgez: []}\n", "graph.edgez"),
    ],
)
def test_an_unknown_key_of_the_file_is_a_usage_error_naming_it(
    no_named_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workflow: str,
    refused: str,
) -> None:
    (no_named_file / "workflow.yaml").write_text(workflow, encoding="utf-8")
    monkeypatch.setattr(qp_serve.uvicorn, "run", never_serve)

    with pytest.raises(SystemExit) as exited:
        qp_serve.main([])

    assert exited.value.code == 2
    assert refused in capsys.readouterr().err


def never_serve(app: object, **options: object) -> None:
    raise AssertionError(f"a server started with {options}: nothing was refused")


def test_a_rerank_size_above_the_retrieve_size_is_no_longer_refused(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stage type knows only its own settings: the reranker keeps what it has.
    monkeypatch.setenv("QP_GRAPH__STAGES__RERANK__TOP_K", "30")
    reranker = RecordingReranker()

    body = ask(settings_factory(**FAKE_BACKENDS), reranker=reranker)

    assert reranker.calls == [(QUESTION, 20, 30)]
    assert len(body["passages"]) == 20


# --- retrieval mode and prompts are the stages' ------------------------------------


def test_lexical_retrieval_is_a_workflow_file_with_no_embed_stage(
    settings_factory: SettingsFactory, no_named_file: Path
) -> None:
    # The fixture's components: ingestion still embeds the passages.
    lexical_graph = yaml.safe_load(LEXICAL_WORKFLOW)["graph"]
    write_workflow_file(no_named_file / "workflow.yaml", lexical_graph)
    store, embedder = ScriptedStore(["a"], ["z", "y"]), RecordingEmbedder()
    ports = {"vector_store": store, "embedder": embedder}

    body = ask(settings_factory(**FAKE_BACKENDS), **ports, reranker=RecordingReranker())

    assert store.calls == [("lexical", QUESTION, 20)]
    assert embedder.queries == []
    assert "embed" not in body["timings_ms"]  # the timings follow the graph
    assert [passage["id"] for passage in body["passages"]] == ["z", "y"]


def test_a_lexical_retrieve_stage_after_an_embed_stage_runs_it_for_nothing(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_GRAPH__STAGES__RETRIEVE__RETRIEVAL_MODE", "lexical")
    # The retrieve stage reads what its retrieval mode needs: words, not a vector.
    monkeypatch.setenv(
        "QP_GRAPH__STAGES__RETRIEVE__READS", '["standalone_query", "metadata_filter"]'
    )
    store, embedder = ScriptedStore(["a"], ["z"]), RecordingEmbedder()

    ask(settings_factory(**FAKE_BACKENDS), vector_store=store, embedder=embedder)

    assert store.calls == [("lexical", QUESTION, 20)]
    assert embedder.queries == [QUESTION]  # the fixture's graph still embeds


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("QP_RETRIEVAL_MODE", "lexical"),
        ("QP_TOP_K_RERANK", "3"),
        ("QP_USER_PROMPT", "no placeholder"),
        ("QP_PROMPTS_FILE", "/nowhere/prompts.yaml"),
    ],
)
def test_the_flat_variables_of_the_stages_are_no_longer_read(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)
    embedder, reranker = RecordingEmbedder(), RecordingReranker()

    body = ask(
        settings_factory(**FAKE_BACKENDS),
        "/query",
        embedder=embedder,
        reranker=reranker,
    )

    assert (embedder.queries, reranker.calls) == ([QUESTION], [(QUESTION, 20, 10)])
    assert body["answer"].startswith("[fake-llm] Context:\n[1] ")
