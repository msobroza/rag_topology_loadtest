"""The RAG's workflow file: it loads, its graph builds, and the workflow app answers.

The app is built from the settings qp-serve builds, from this directory, where
it finds ``workflow.yaml`` with no flag, and driven through its HTTP API. Every
component is moved to its fake backend by the variable its path names, as on a
laptop: no weights, no store, no network. What the file pins (the RAG's models,
store, sizes, prompts and graph) is read back as the settings read it, the one
place it is written since the toolkit defaults none of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from query_processing.application.graph_check import check_graph
from query_processing.composition.composition_root import CompositionRoot
from query_processing.composition.registered_names import check_registered_names
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app

QUESTION = "How many paid vacation days do employees accrue each year?"
FAKE_BACKENDS = {
    f"QP_COMPONENTS__{name}__BACKEND": "fake"
    for name in ("EMBEDDER", "RERANKER", "LLM", "VECTOR_STORE")
}
STAGES = ["embed", "retrieve", "rerank", "generate"]
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite passages as [n]. If the context is insufficient, say so."
)
USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


def settings_here() -> WorkflowSettings:
    """The settings qp-serve builds here, refused as it refuses them at start-up."""
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]
    check_registered_names(settings)
    check_graph(settings.graph, settings.component_types())
    return settings


@pytest.fixture
def all_fake(use_case_dir: Path, monkeypatch: pytest.MonkeyPatch) -> WorkflowSettings:
    """This directory's workflow file, every component on its fake backend."""
    for name, value in FAKE_BACKENDS.items():
        monkeypatch.setenv(name, value)
    return settings_here()


@pytest.mark.parametrize("path", ["/retrieve", "/query"])
def test_the_workflow_file_builds_a_graph_that_answers(
    all_fake: WorkflowSettings, path: str
) -> None:
    with TestClient(create_app(CompositionRoot(all_fake))) as app:
        response = app.post(path, json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    # One timing per stage of the file, in its order, then the total.
    assert list(body["timings_ms"]) == [*STAGES, "total"]
    assert len(body["passages"]) == 10  # what the rerank stage keeps
    run_label = "embedder=fake+reranker=fake+llm=fake+vector_store=fake"
    assert response.headers["X-RAG-Run-Label"] == run_label
    if path == "/query":  # the fake LLM echoes the file's user prompt, filled
        assert body["answer"].startswith("[fake-llm] Context:\n[1] ")
    else:
        assert body["answer"] is None


def test_the_file_pins_the_rags_models_its_endpoint_and_its_store(
    use_case_dir: Path,
) -> None:
    components = settings_here().components

    assert list(components) == ["embedder", "reranker", "llm", "vector_store"]
    embedder, reranker = components["embedder"], components["reranker"]
    assert (embedder.backend, embedder.source_id, embedder.alias) == (
        "local",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "snowflake-arctic-embed-l-v2.0",
    )
    assert embedder.query_prefix == "query: "  # what arctic-embed expects
    assert (reranker.backend, reranker.source_id, reranker.alias) == (
        "local",
        "antoinelouis/crossencoder-camembert-base-mmarcoFR",
        "crossencoder-camembert-base-mmarcoFR",
    )
    llm = components["llm"]
    assert (llm.backend, llm.api.url, llm.api.api_format, llm.api.model) == (
        "api",
        "https://api.openai.com/v1/chat/completions",
        "openai",
        "gpt-4o-mini",
    )
    assert llm.http.timeout_s == 120.0  # an answer takes longer than a model call
    store = components["vector_store"]
    assert (store.backend, store.elasticsearch.url) == (
        "elasticsearch",
        "http://localhost:9200",
    )
    assert (store.elasticsearch.index, store.chroma.collection) == (
        "rag_passages",
        "rag_passages",
    )


def test_the_file_pins_the_rags_graph_its_sizes_and_its_prompts(
    use_case_dir: Path,
) -> None:
    graph = settings_here().graph

    stages = {name: stage.stage_settings for name, stage in graph.stages.items()}
    assert stages == {
        "embed": {},
        "retrieve": {"retrieval_mode": "dense", "top_k": 20},
        "rerank": {"top_k": 10},
        "generate": {"system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT},
    }
    calls = {name: stage.call for name, stage in graph.stages.items()}
    assert calls == {
        "embed": "embedder",
        "retrieve": "vector_store",
        "rerank": "reranker",
        "generate": "llm",
    }
    lists = {name: (s.reads, s.writes) for name, s in graph.stages.items()}
    assert lists == {
        "embed": (["standalone_query"], ["query_embedding"]),
        "retrieve": (["query_embedding", "metadata_filter"], ["candidates"]),
        "rerank": (["standalone_query", "candidates"], ["reranked"]),
        "generate": (["conversational_query", "reranked"], ["answer"]),
    }
    assert graph.edge_pairs() == [
        ("START", "embed"),
        ("embed", "retrieve"),
        ("retrieve", "rerank"),
        ("generate", "END"),
    ]
    (branch,) = graph.conditional_edges
    assert (branch.source, branch.path) == ("rerank", "is_query_mode")
    assert branch.path_map == {True: "generate", False: "END"}
    assert branch.reads == ["mode"]


def test_the_load_test_sends_a_model_server_the_sizes_of_the_files_stages(
    use_case_dir: Path,
) -> None:
    # The toolkit defaults neither: the environment example states the RAG's.
    example = (use_case_dir / ".env.example").read_text(encoding="utf-8")
    sizes = dict(re.findall(r"^(QP_TOP_K_\w+)=(\d+)", example, re.M))
    stages = settings_here().graph.stages

    assert sizes == {
        "QP_TOP_K_RETRIEVE": str(stages["retrieve"].stage_settings["top_k"]),
        "QP_TOP_K_RERANK": str(stages["rerank"].stage_settings["top_k"]),
    }
