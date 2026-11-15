"""``RagSettings``: the tunables, their validation, and the graph of the workflow file."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_graphs import fixture_graph, write_workflow_file
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import WORKFLOW_FILE_VARIABLE

SettingsFactory = Callable[..., RagSettings]
SYSTEM_PROMPT = "Answer from the numbered fixture passages only."  # the fixture's
USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


def stage_settings(settings: RagSettings) -> dict[str, dict[str, object]]:
    """Each stage's own settings, by stage name."""
    return {name: stage.stage_settings for name, stage in settings.graph.stages.items()}


def local_threads(settings: RagSettings) -> tuple[int, int]:
    """The embedder's and the reranker's thread counts, from their local blocks."""
    components = settings.components
    embedder, reranker = components["embedder"], components["reranker"]
    return embedder.local.threads, reranker.local.threads


# --- tunables ------------------------------------------------------------------


def test_tunable_defaults(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    store = s.components["vector_store"]
    # The index is the file's: where the passages are kept has no default.
    assert (store.elasticsearch.index, store.fake.seed_documents) == (
        "fixture_passages",
        200,
    )
    assert s.local_models.workers == 4
    assert (s.calibration.budget_s, s.calibration.rounds) == (90.0, 1)
    assert (s.http.timeout_s, s.http.retries, s.http.backoff_s) == (30.0, 2, 0.2)
    assert s.domino.token_url == "http://localhost:8899/access-token"
    assert s.domino.token_ttl_s == 240.0


def test_tunables_read_rag_prefixed_env(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {
        "RAG_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__INDEX": "passages_v2",
        "RAG_COMPONENTS__VECTOR_STORE__FAKE__SEED_DOCUMENTS": "30",
        "RAG_LOCAL_MODELS__WORKERS": "2",
        "RAG_HTTP__TIMEOUT_S": "5.5",
        "RAG_HTTP__RETRIES": "0",
        "RAG_HTTP__BACKOFF_S": "0.5",
        "RAG_DOMINO__TOKEN_URL": "http://token.test/t",
        "RAG_DOMINO__TOKEN_TTL_S": "60",
        "RAG_CALIBRATION__BUDGET_S": "12.5",
        "RAG_CALIBRATION__ROUNDS": "3",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    s = settings_factory()
    store = s.components["vector_store"]
    assert (store.elasticsearch.index, store.fake.seed_documents) == ("passages_v2", 30)
    assert s.local_models.workers == 2
    assert (s.http.timeout_s, s.http.retries, s.http.backoff_s) == (5.5, 0, 0.5)
    assert (s.domino.token_url, s.domino.token_ttl_s) == ("http://token.test/t", 60.0)
    assert (s.calibration.budget_s, s.calibration.rounds) == (12.5, 3)


@pytest.mark.parametrize(
    "field",
    [
        "components__reranker__local__calibration_passages",
        "components__vector_store__fake__seed_documents",
        "local_models__workers",
    ],
)
def test_counts_below_one_are_rejected(
    settings_factory: SettingsFactory, field: str
) -> None:
    with pytest.raises(ValidationError, match=field.split("__")[-1]):
        settings_factory(**{field: 0})


# --- the workflow file -------------------------------------------------------------


def test_the_fixture_file_reads_into_stages_edges_and_a_conditional_edge(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory()

    assert stage_settings(s) == {
        "embed": {},
        "retrieve": {"retrieval_mode": "dense", "top_k": 20},
        "rerank": {"top_k": 10},
        "generate": {"system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT},
    }
    calls = {name: stage.call for name, stage in s.graph.stages.items()}
    assert calls == {
        "embed": "embedder",
        "retrieve": "vector_store",
        "rerank": "reranker",
        "generate": "llm",
    }
    assert s.graph.edge_pairs() == [
        ("START", "embed"),
        ("embed", "retrieve"),
        ("retrieve", "rerank"),
        ("generate", "END"),
    ]
    (branch,) = s.graph.conditional_edges
    assert (branch.source, branch.path) == ("rerank", "is_query_mode")
    assert branch.path_map == {True: "generate", False: "END"}
    # The type of each stage is its name.
    assert all(stage.type is None for stage in s.graph.stages.values())


def test_a_variable_overrides_one_prompt_and_keeps_the_other(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_GRAPH__STAGES__GENERATE__SYSTEM_PROMPT", "Be terse.")

    generate = stage_settings(settings_factory())["generate"]

    assert generate == {"system_prompt": "Be terse.", "user_prompt": USER_PROMPT}


def test_the_sources_win_in_their_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One key per source, each set by that source and every source below it.
    graph = fixture_graph()
    graph["stages"]["retrieve"].update(top_k=7, rrf_k=61)
    graph["stages"]["rerank"]["top_k"] = 5
    workflow = write_workflow_file(tmp_path / "workflow.yaml", graph)
    with workflow.open("a", encoding="utf-8") as file:
        file.write("local_models: {openvino_dir: from_the_file}\n")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "RAG_GRAPH__STAGES__RETRIEVE__TOP_K=8\n"
        "RAG_GRAPH__STAGES__RERANK__TOP_K=6\n"
        "RAG_GRAPH__STAGES__RETRIEVE__RRF_K=62\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_GRAPH__STAGES__RERANK__TOP_K", "4")
    monkeypatch.setenv("RAG_GRAPH__STAGES__RETRIEVE__RRF_K", "63")

    s = RagSettings(
        _env_file=dotenv,  # type: ignore[call-arg]
        graph={"stages": {"retrieve": {"rrf_k": 64}}},
    )

    retrieve, rerank = stage_settings(s)["retrieve"], stage_settings(s)["rerank"]
    assert int(retrieve["rrf_k"]) == 64  # constructor > environment
    assert int(rerank["top_k"]) == 4  # environment > .env
    assert int(retrieve["top_k"]) == 8  # .env > the workflow file
    # The workflow file > the class default, which keeps the block's other keys.
    assert s.local_models.openvino_dir == "from_the_file"
    assert s.local_models.workers == 4  # the class default
    assert retrieve["retrieval_mode"] == "dense"  # each source sets only its keys


def test_an_unknown_top_level_key_of_the_file_is_refused_naming_it(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = write_workflow_file(tmp_path / "workflow.yaml", fixture_graph())
    with workflow.open("a", encoding="utf-8") as file:
        file.write("top_k_retrieve: 20\n")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    with pytest.raises(ValueError, match=r"unknown key.*'top_k_retrieve'"):
        settings_factory()


def test_an_unknown_key_of_the_graph_is_refused_naming_it(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_GRAPH__STAGEZ__RERANK__TOP_K", "3")

    with pytest.raises(ValidationError, match=r"graph\.stagez"):
        settings_factory()


def test_a_file_without_a_graph_is_refused_naming_it(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("http: {retries: 0}\n", encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    with pytest.raises(ValidationError, match="graph"):
        settings_factory()


# --- thread budget inputs ---------------------------------------------------------


def test_thread_budget_inputs_default_to_find_out(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY", raising=False)
    monkeypatch.delenv("DOMINO_HARDWARE_TIER_ID", raising=False)

    s = settings_factory()
    assert (s.local_models.cpus, *local_threads(s)) == (0, 0, 0)
    assert (s.domino_api_proxy, s.domino_hardware_tier_id) == ("", "")


def test_thread_budget_inputs_read_rag_env_and_domino_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LOCAL_MODELS__CPUS", "2.5")
    monkeypatch.setenv("RAG_COMPONENTS__EMBEDDER__LOCAL__THREADS", "3")
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__LOCAL__THREADS", "1")
    monkeypatch.setenv("DOMINO_API_PROXY", "http://localhost:8899")
    monkeypatch.setenv("DOMINO_HARDWARE_TIER_ID", "small-k8s")

    s = settings_factory()
    assert (s.local_models.cpus, *local_threads(s)) == (2.5, 3, 1)
    assert s.domino_api_proxy == "http://localhost:8899"
    assert s.domino_hardware_tier_id == "small-k8s"


def test_domino_variables_are_not_read_with_the_rag_prefix(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY", raising=False)
    monkeypatch.setenv("RAG_DOMINO_API_PROXY", "http://wrong")

    assert settings_factory().domino_api_proxy == ""


def test_domino_variables_can_be_passed_by_field_name(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(domino_api_proxy="http://p", domino_hardware_tier_id="t")
    assert (s.domino_api_proxy, s.domino_hardware_tier_id) == ("http://p", "t")


@pytest.mark.parametrize(
    "field",
    [
        "local_models__cpus",
        "components__embedder__local__threads",
        "components__reranker__local__threads",
    ],
)
def test_negative_thread_budget_inputs_are_rejected(
    settings_factory: SettingsFactory, field: str
) -> None:
    with pytest.raises(ValidationError, match=field.split("__")[-1]):
        settings_factory(**{field: -1})


def test_each_value_names_the_variable_it_is_read_from() -> None:
    path = ("components", "reranker", "local", "threads")
    assert RagSettings.variable_of(*path) == "RAG_COMPONENTS__RERANKER__LOCAL__THREADS"
    assert RagSettings.variable_of("local_models", "cpus") == "RAG_LOCAL_MODELS__CPUS"
    assert RagSettings.variable_of("domino_api_proxy") == "DOMINO_API_PROXY"


def test_a_field_that_does_not_exist_names_no_variable() -> None:
    with pytest.raises(ValueError, match="'rerank_threads'"):
        RagSettings.variable_of("rerank_threads")
