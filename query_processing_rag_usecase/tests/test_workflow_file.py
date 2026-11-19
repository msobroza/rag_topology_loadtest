"""The RAG's workflow file: it loads, its graph builds, and the workflow app answers.

The app is built from the settings qp-serve builds, from this directory, where
it finds ``workflow.yaml`` with no flag, and driven through its HTTP API. Every
component is moved to its fake backend by the variable its path names, as on a
laptop: no weights, no pipeline file, no store, no network. What the file pins
(the RAG's extension, models, router, store, sizes, prompts and graph) is read
back as the settings read it, the one place it is written since the toolkit
defaults none of it. The paths its graph takes are in test_conversation_paths.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from query_processing.application.prompt_rendering import render_prompt
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.conversation import HistoryMessage
from query_processing.domain.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
)
from query_processing.domain.retrieval.passages import ScoredPassage
from query_processing.entrypoints.qp_serve import create_app
from query_processing.testing import registered_orchestrators

from query_processing_rag_usecase.config.knn_router_settings import KnnRouterSettings
from workflow_on_fakes import (
    COMPONENTS,
    FAKE_RUN_LABEL,
    all_fake_settings,
    settings_here,
)

QUESTION = "How many paid vacation days do employees accrue each year?"
EARLIER_TURNS = [
    {"role": "user", "content": "What does the handbook say about vacation?"},
    {"role": "assistant", "content": "Fifteen days a year."},
]
STAGES = [
    "embed_for_route",
    "route",
    "rewrite",
    "embed",
    "retrieve",
    "rerank",
    "generate",
]
# The references of the corpus: the values a condition on `source` can take.
SOURCES = [
    "A-25",
    "AMF",
    "CHAD",
    "GAA",
    "code_securite_routiere",
    "educaloi",
    "infoassurances",
]
# The generate stage's two settings before prompts.yaml, which its answer
# prompt renders character for character.
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite passages as [n]. If the context is insufficient, say so."
)
USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"
# What the rewrite prompt's system message renders for the file's store and a
# conversational history whose first message was restricted to one source.
REWRITE_SYSTEM = (
    "Rewrite the last user message so that it needs no conversation to be "
    "understood.\n"
    "Add a condition on a metadata field only when the conversation names one "
    "of its values.\n"
    "- source: The reference the passage comes from. Values: A-25, AMF, CHAD, "
    "GAA, code_securite_routiere, educaloi, infoassurances\n"
    "An earlier message was restricted to: "
    '[{"attribute": "source", "comparator": "in", "value": ["AMF"]}]\n'
)


def written_workflow(use_case_dir: Path) -> dict[str, Any]:
    """The workflow file as it is written: its keys, before any default."""
    text = (use_case_dir / "workflow.yaml").read_text(encoding="utf-8")
    written: dict[str, Any] = yaml.safe_load(text)
    return written


@pytest.mark.parametrize("path", ["/retrieve", "/query"])
@pytest.mark.parametrize(
    "earlier", [[], EARLIER_TURNS], ids=["first-message", "follow-up"]
)
def test_the_workflow_file_builds_a_graph_that_answers(
    all_fake: WorkflowSettings, path: str, earlier: list[dict[str, str]]
) -> None:
    body = {"question": QUESTION, "conversational_history": earlier}
    with TestClient(create_app(CompositionRoot(all_fake))) as app:
        response = app.post(path, json=body)

    assert response.status_code == 200, response.text
    answered = response.json()
    # One timing per stage of the file, in its order, then the total.
    assert list(answered["timings_ms"]) == [*STAGES, "total"]
    assert len(answered["passages"]) == 10  # what the rerank stage keeps
    assert response.headers["X-RAG-Run-Label"] == FAKE_RUN_LABEL
    if path == "/query":  # the fake LLM echoes the file's user prompt, filled
        assert answered["answer"].startswith("[fake-llm] Context:\n[1] ")
    else:
        assert answered["answer"] is None


@pytest.mark.parametrize("orchestrator", registered_orchestrators())
def test_every_orchestrator_accepts_the_files_graph_at_start_up(
    use_case_dir: Path, monkeypatch: pytest.MonkeyPatch, orchestrator: str
) -> None:
    # all_fake_settings refuses the settings as qp-serve does at start-up:
    # the workflow-graph check asks the chosen orchestrator about the graph.
    settings = all_fake_settings(monkeypatch, QP_ORCHESTRATOR=orchestrator)

    assert settings.orchestrator == orchestrator
    # The file names none: the same file runs on each, LangGraph by default.
    assert "orchestrator" not in written_workflow(use_case_dir)


def test_the_file_names_its_extension_and_writes_the_runtime(
    use_case_dir: Path,
) -> None:
    settings = settings_here()

    assert settings.extensions == ["query_processing_rag_usecase"]
    assert written_workflow(use_case_dir)["local_models"] == {"runtime": "openvino"}
    assert settings.local_models.runtime == "openvino"


def test_the_file_pins_the_rags_models_and_its_endpoint(use_case_dir: Path) -> None:
    components = settings_here().components

    assert list(components) == list(COMPONENTS)
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


def test_the_file_pins_the_router_its_embedder_and_its_training_values(
    use_case_dir: Path,
) -> None:
    components = settings_here().components

    solon, router = components["router_embedder"], components["router"]
    assert (solon.type, solon.backend, solon.source_id, solon.alias) == (
        "embedder",
        "local",
        "OrdalieTech/Solon-embeddings-large-0.1",
        "Solon-embeddings-large-0.1",
    )
    assert solon.query_prefix == "query : "  # what Solon expects, its space included
    assert (router.type, router.backend, router.embedder) == (
        "router",
        "local",
        "router_embedder",
    )
    assert router.routes == ["self_contained", "leans_on_history"]
    assert router.backend_block() == KnnRouterSettings(
        pipeline_file="router_pipeline.joblib", n_neighbors=3, metric="cosine"
    )


def test_the_file_states_one_store_its_source_field_and_no_prefilter(
    use_case_dir: Path,
) -> None:
    store = settings_here().components["vector_store"]
    written = written_workflow(use_case_dir)["components"]["vector_store"]

    # The block of the backend in use and the declared field, nothing else: no
    # block of another store, and no prefilter, which is a site's (deploy.yaml).
    assert set(written) == {
        "backend",
        "elasticsearch",
        "metadata_field_info",
    }
    assert (store.backend, store.elasticsearch.url, store.elasticsearch.index) == (
        "elasticsearch",
        "http://localhost:9200",
        "rag_passages",
    )
    assert store.prefilter == []
    assert store.metadata_field_info == [
        MetadataFieldInfo(
            name="source",
            type="string",
            description="The reference the passage comes from.",
            allowed_values=SOURCES,
        )
    ]


def test_the_file_pins_the_rags_stages_their_sizes_and_their_prompts(
    use_case_dir: Path,
) -> None:
    stages = settings_here().graph.stages

    assert list(stages) == STAGES
    assert {name: stage.stage_settings for name, stage in stages.items()} == {
        "embed_for_route": {},
        "route": {},
        "rewrite": {
            "prompt": "rewrite_as_standalone",
            "output_schema": "standalone_query_with_filter",
            "metadata_fields_of": "vector_store",
        },
        "embed": {},
        "retrieve": {"retrieval_mode": "dense", "top_k": 20},
        "rerank": {"top_k": 10},
        "generate": {"prompt": "answer_from_context"},
    }
    assert {name: (stage.type, stage.call) for name, stage in stages.items()} == {
        "embed_for_route": (None, "router_embedder"),
        "route": (None, "router"),
        "rewrite": ("generate", "llm"),
        "embed": (None, "embedder"),
        "retrieve": (None, "vector_store"),
        "rerank": (None, "reranker"),
        "generate": (None, "llm"),
    }
    assert {name: (s.reads, s.writes) for name, s in stages.items()} == {
        "embed_for_route": (["conversational_query"], ["routing_embedding"]),
        "route": (["routing_embedding"], ["route"]),
        "rewrite": (
            ["conversational_query", "conversational_history"],
            ["standalone_query", "metadata_filter"],
        ),
        "embed": (["standalone_query"], ["query_embedding"]),
        "retrieve": (["query_embedding", "metadata_filter"], ["candidates"]),
        "rerank": (["standalone_query", "candidates"], ["reranked"]),
        "generate": (["conversational_query", "reranked"], ["answer"]),
    }


def test_the_file_pins_the_rags_edges_and_its_three_conditional_edges(
    use_case_dir: Path,
) -> None:
    graph = settings_here().graph

    assert graph.edge_pairs() == [
        ("embed_for_route", "route"),
        ("rewrite", "embed"),
        ("embed", "retrieve"),
        ("retrieve", "rerank"),
        ("generate", "END"),
    ]
    branches = [
        (b.source, b.path, b.reads, b.path_map) for b in graph.conditional_edges
    ]
    assert branches == [
        (
            "START",
            "has_history",
            ["conversational_history"],
            {True: "embed_for_route", False: "embed"},
        ),
        (
            "route",
            "label",
            ["route"],
            {"self_contained": "embed", "leans_on_history": "rewrite"},
        ),
        ("rerank", "is_query_mode", ["mode"], {True: "generate", False: "END"}),
    ]


def old_messages(question: str, passages: list[ScoredPassage]) -> list[dict[str, str]]:
    """What the generate stage's two settings gave before prompts.yaml."""
    context = "\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, start=1))
    user = USER_PROMPT.format(context=context, question=question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_the_answer_prompt_renders_what_the_two_old_settings_gave(
    use_case_dir: Path, count: int
) -> None:
    prompt = settings_here().prompts["answer_from_context"]
    passages = [
        ScoredPassage(id=str(i), text=f"Passage {{x}} {i}.", retrieval_score=1.0)
        for i in range(count)
    ]

    messages = render_prompt(
        prompt, {"reranked": passages, "conversational_query": "Why {y}?"}
    )

    assert messages == old_messages("Why {y}?", passages)


def test_the_rewrite_prompt_lists_the_stores_field_and_splices_the_earlier_turns(
    use_case_dir: Path,
) -> None:
    settings = settings_here()
    declared = settings.components["vector_store"].metadata_field_info
    on_amf = MetadataCondition(attribute="source", comparator="in", value=["AMF"])
    earlier = [
        HistoryMessage(
            role="user",
            content="Que dit l'AMF sur la franchise ?",
            metadata_filter=[on_amf],
        ),
        HistoryMessage(role="assistant", content="Elle est de 500 $."),
    ]

    messages = render_prompt(
        settings.prompts["rewrite_as_standalone"],
        {
            "metadata_field_info": declared,
            "conversational_history": earlier,
            "conversational_query": "Et pour un véhicule loué ?",
        },
    )

    assert messages == [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": "Que dit l'AMF sur la franchise ?"},
        {"role": "assistant", "content": "Elle est de 500 $."},
        {"role": "user", "content": "Et pour un véhicule loué ?"},
    ]


def test_the_rewrite_prompt_lists_a_declared_field_without_allowed_values(
    use_case_dir: Path,
) -> None:
    # A field may leave its allowed values out: the prompt lists it with none.
    settings = settings_here()
    (source,) = settings.components["vector_store"].metadata_field_info
    year = source.model_copy(
        update={"name": "year", "description": "The year.", "allowed_values": None}
    )

    messages = render_prompt(
        settings.prompts["rewrite_as_standalone"],
        {
            "metadata_field_info": [source, year],
            "conversational_history": [],
            "conversational_query": "Et en 2020 ?",
        },
    )

    lines = messages[0]["content"].splitlines()
    assert lines[2].startswith("- source: The reference the passage comes from.")
    assert lines[3] == "- year: The year."


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


def test_the_environment_example_shows_the_two_new_components_on_fake(
    use_case_dir: Path,
) -> None:
    # The first run, with no model weights, needs both on fake.
    example = (use_case_dir / ".env.example").read_text(encoding="utf-8")

    on_fake = re.findall(r"^# (QP_COMPONENTS__\w+__BACKEND)=fake\b", example, re.M)

    assert "QP_COMPONENTS__ROUTER_EMBEDDER__BACKEND" in on_fake
    assert "QP_COMPONENTS__ROUTER__BACKEND" in on_fake
