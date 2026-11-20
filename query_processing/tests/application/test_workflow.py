"""The RAG graph of the tests' workflow file, built over fake ports.

Each test that builds the workflow runs on every registered orchestrator (the
``orchestrator`` fixture of conftest.py), but the one that reaches LangGraph's
own graph for the names a tracer is given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel
from query_processing_core.retrieval.metadata_filter import MetadataCondition

from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.orchestrator.base_orchestrator import BaseOrchestrator
from query_processing.adapters.reranker.none.no_reranker import NoReranker
from query_processing.application.rag_dependencies import RagDependencies
from query_processing.application.stages.base_stage import StageContext
from query_processing.application.workflow import (
    CompiledWorkflow,
    stage_timings,
    stage_tokens,
)
from query_processing.application.workflow_state import initial_state
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.components.component_requests import GenerateAnswer
from query_processing.domain.graph_description import GraphDescription
from query_processing.domain.prompts import Prompt, PromptMessage
from query_processing.domain.retrieval.corpus import synthetic_questions
from query_processing.domain.retrieval.rag_result import RagMode
from rag_graphs import fixture_graph, rag_graph
from workflow_fakes import (
    ON_LANGGRAPH_ALONE,
    ChainNameRecorder,
    DenseOnlyStore,
    RecordingEmbedder,
    RecordingReranker,
    ScriptedStore,
    build_workflow_on,
    compiled_langgraph,
)

VACATION_QUESTION = synthetic_questions(1)[0]
# The four stages in the graph's order, then the total: the keys of today's response.
TIMING_KEYS = ["embed", "retrieve", "rerank", "generate", "total"]
DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_K_RERANK = 20, 10
WorkflowFor = Callable[..., CompiledWorkflow]
# A prompt other than the fixture's: the question, then the passages.
# A generate stage before the fixture's: it rewrites the question it searches with.
REWRITE_STAGE = {
    "type": "generate",
    "call": "llm",
    "reads": ["conversational_query"],
    "writes": ["standalone_query"],
    "prompt": "rewrite",
}
REWRITE_PROMPT = [
    PromptMessage(role="user", content="Rewrite: {{ conversational_query }}")
]
ASKED_USING = "ASK {{ conversational_query }} USING {% for p in reranked %}[{{ loop.index }}] {{ p.text }} {% endfor %}"


@pytest.fixture
def workflow_for(
    fake_settings: WorkflowSettings, orchestrator: type[BaseOrchestrator]
) -> WorkflowFor:
    """Today's graph with ``changes`` (see ``rag_graph``), over ``deps``, on ``orchestrator``.

    Its prompts are the fixture's, and ``prompts`` adds or replaces some.
    """

    def make(
        deps: RagDependencies, prompts: dict[str, Prompt] | None = None, **changes: Any
    ) -> CompiledWorkflow:
        context = StageContext({**fake_settings.prompts, **(prompts or {})})
        return build_workflow_on(
            orchestrator, rag_graph(**changes), deps.components(), context
        )

    return make


def with_ports(deps: RagDependencies, **ports: object) -> RagDependencies:
    """``deps`` with some components replaced, by name."""
    return RagDependencies({**deps.by_name, **ports})


def scripted_deps(
    fake_deps: RagDependencies, store: ScriptedStore, embedder: RecordingEmbedder
) -> RagDependencies:
    return with_ports(
        fake_deps, vector_store=store, embedder=embedder, reranker=NoReranker()
    )


# --- the graph ----------------------------------------------------------------


async def test_query_mode_answers_from_reranked_passages(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    assert result.question == VACATION_QUESTION
    assert result.mode == "query"
    assert result.answer is not None
    assert result.answer.startswith("[fake-llm]")
    assert len(result.passages) == DEFAULT_TOP_K_RERANK
    scores = [p.rerank_score for p in result.passages]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


async def test_query_mode_times_every_stage(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    assert list(result.timings_ms) == TIMING_KEYS
    assert result.timings_ms["total"] > 0
    assert all(ms >= 0 for ms in result.timings_ms.values())


async def test_vacation_question_retrieves_vacation_policy(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    topics = [p.metadata["topic"] for p in result.passages]
    # One seeded document in ten is a vacation policy; they outrank every other topic.
    assert topics[:3] == ["vacation-policy"] * 3


async def test_retrieve_mode_skips_generation(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION, mode="retrieve")

    assert isinstance(fake_deps.by_name["llm"], FakeLLM)
    assert fake_deps.by_name["llm"].calls == 0
    assert result.mode == "retrieve"
    assert result.answer is None
    assert list(result.timings_ms) == TIMING_KEYS
    assert result.timings_ms["generate"] == 0.0
    assert result.timings_ms["total"] > 0
    assert len(result.passages) == DEFAULT_TOP_K_RERANK


async def test_a_query_reports_the_tokens_of_its_answer_under_its_stage(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    assert list(result.tokens) == ["generate"]
    assert set(result.tokens["generate"]) == {"prompt", "completion"}
    assert result.tokens["generate"]["completion"] > 0


async def test_a_retrieve_reports_no_tokens(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION, mode="retrieve")

    assert result.tokens == {}


class UncountedLLM(FakeLLM):
    """The fake LLM, whose endpoint reports one count of two, or none."""

    def __init__(self, prompt_tokens: int | None) -> None:
        super().__init__()
        self.prompt_tokens = prompt_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        return GenerateAnswer(text="an answer", prompt_tokens=self.prompt_tokens)


@pytest.mark.parametrize("prompt_tokens", [12, None], ids=["one-count", "none"])
async def test_an_answer_without_both_counts_reports_no_tokens(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, prompt_tokens: int | None
) -> None:
    deps = with_ports(fake_deps, llm=UncountedLLM(prompt_tokens))

    result = await workflow_for(deps).run(VACATION_QUESTION)

    assert (result.answer, result.tokens) == ("an answer", {})


def rewritten_first() -> GraphDescription:
    """The fixture's graph, with a second generate stage that rewrites the question."""
    graph = fixture_graph()
    graph["stages"] = {"rewrite": REWRITE_STAGE, **graph["stages"]}
    graph["edges"] = [["START", "rewrite", "embed"], *graph["edges"]]
    graph["edges"][1] = [n for n in graph["edges"][1] if n != "START"]
    return GraphDescription.model_validate(graph)


@pytest.mark.parametrize(
    ("mode", "reported"),
    [("query", ["rewrite", "generate"]), ("retrieve", ["rewrite"])],
)
async def test_each_generate_stage_that_runs_reports_its_tokens_under_its_name(
    fake_deps: RagDependencies,
    fake_settings: WorkflowSettings,
    orchestrator: type[BaseOrchestrator],
    mode: RagMode,
    reported: list[str],
) -> None:
    context = StageContext({**fake_settings.prompts, "rewrite": REWRITE_PROMPT})
    workflow = build_workflow_on(
        orchestrator, rewritten_first(), fake_deps.components(), context
    )

    result = await workflow.run("How many vacation days?", mode=mode)

    assert list(result.tokens) == reported  # in the graph's order
    # The fake's word counts: "Rewrite: How many vacation days?", then its echo.
    assert result.tokens["rewrite"] == {"prompt": 5, "completion": 6}


async def test_no_reranker_keeps_retrieval_order(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    deps = with_ports(fake_deps, reranker=NoReranker())
    result = await workflow_for(deps).run(VACATION_QUESTION, mode="retrieve")

    vector = await deps.by_name["embedder"].embed_query(VACATION_QUESTION)
    expected = await deps.by_name["vector_store"].query(vector, DEFAULT_TOP_K_RETRIEVE)
    first_kept = [p.id for p in expected[:DEFAULT_TOP_K_RERANK]]
    assert [p.id for p in result.passages] == first_kept
    assert all(p.rerank_score is None for p in result.passages)


async def test_non_default_settings_size_retrieval_rerank_and_prompt(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    deps = with_ports(fake_deps, reranker=NoReranker())
    workflow = workflow_for(
        deps,
        retrieve_top_k=3,
        rerank_top_k=2,
        prompt="asked",
        prompts={"asked": [PromptMessage(role="user", content=ASKED_USING)]},
    )

    result = await workflow.run("vacation days")

    assert len(result.passages) == 2
    assert result.answer is not None
    # FakeLLM echoes the first 200 characters of the user message.
    assert result.answer.startswith("[fake-llm] ASK vacation days USING [1] ")


async def test_the_reranker_gets_the_question_the_retrieve_size_and_its_own(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    reranker = RecordingReranker()
    deps = with_ports(fake_deps, reranker=reranker)
    workflow = workflow_for(deps, retrieve_top_k=7, rerank_top_k=4)

    await workflow.run(VACATION_QUESTION, mode="retrieve")

    assert reranker.calls == [(VACATION_QUESTION, 7, 4)]


def test_the_components_are_those_the_stages_call_by_their_names(
    fake_deps: RagDependencies,
) -> None:
    ports = fake_deps.by_name
    assert fake_deps.components() == dict(ports)
    assert set(ports) == {"embedder", "reranker", "llm", "vector_store"}
    # An LLM has no readiness probe; the embedder and the LLM name their model.
    probed = [name for name, _ in fake_deps.ready_probes()]
    assert sorted(probed) == ["embedder", "reranker", "vector_store"]
    assert fake_deps.model_names() == {
        "embedder": ports["embedder"].model_name,
        "llm": ports["llm"].model_name,
    }


@ON_LANGGRAPH_ALONE
@pytest.mark.parametrize("mode", ["query", "retrieve"])
async def test_every_step_of_the_graph_reaches_a_tracer_with_a_name(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, mode: Any
) -> None:
    # LangGraph passes no serialized form and names a step after its function,
    # except a lambda; MLflow's LangChain autolog (Domino tracing) then failed
    # on the unnamed step: "'NoneType' object has no attribute 'get'".
    recorder = ChainNameRecorder()

    graph = compiled_langgraph(workflow_for(fake_deps))
    await graph.ainvoke(
        initial_state(VACATION_QUESTION, mode), {"callbacks": [recorder]}
    )

    assert None not in recorder.names, recorder.names
    assert "is_query_mode" in recorder.names  # the edge condition keeps its name


# --- retrieval modes -------------------------------------------------------------


async def test_dense_mode_asks_the_store_by_vector_only(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a", "b"], ["z"]), RecordingEmbedder()
    deps = scripted_deps(fake_deps, store, embedder)
    workflow = workflow_for(deps, retrieve_top_k=2, rerank_top_k=2)

    result = await workflow.run("vacation days", mode="retrieve")

    assert [p.id for p in result.passages] == ["a", "b"]
    assert store.calls == [("dense", [1.0, 0.0], 2)]


async def test_lexical_mode_skips_the_embed_stage_and_asks_by_words(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a"], ["z", "y"]), RecordingEmbedder()
    deps = scripted_deps(fake_deps, store, embedder)
    workflow = workflow_for(
        deps, retrieval_mode="lexical", retrieve_top_k=2, rerank_top_k=2
    )

    result = await workflow.run("vacation days", mode="retrieve")

    assert [p.id for p in result.passages] == ["z", "y"]
    assert store.calls == [("lexical", "vacation days", 2)]
    assert embedder.queries == []  # no vector is needed, so none is computed
    assert list(result.timings_ms) == ["retrieve", "rerank", "generate", "total"]
    assert result.timings_ms["retrieve"] > 0.0


async def test_hybrid_mode_asks_both_ways_and_fuses_by_rank(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a", "b", "c"], ["b", "d"]), RecordingEmbedder()
    deps = scripted_deps(fake_deps, store, embedder)
    workflow = workflow_for(
        deps, retrieval_mode="hybrid", retrieve_top_k=3, rerank_top_k=3
    )

    result = await workflow.run("vacation days", mode="retrieve")

    # b is in both rankings; a and d each lead one; c is cut by retrieve's top_k.
    assert [p.id for p in result.passages] == ["b", "a", "d"]
    assert sorted(store.calls, key=lambda call: call[0]) == [
        ("dense", [1.0, 0.0], 3),
        ("lexical", "vacation days", 3),
    ]


async def test_the_embed_stage_embeds_the_question_as_a_query(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a"], ["a"]), RecordingEmbedder()
    workflow = workflow_for(
        scripted_deps(fake_deps, store, embedder), retrieval_mode="hybrid"
    )

    result = await workflow.run("vacation days", mode="retrieve")

    # As the user asked it: the embedder puts its own prefix in front.
    assert (embedder.queries, embedder.passages) == (["vacation days"], [])
    lexical_calls = [call for call in store.calls if call[0] == "lexical"]
    assert lexical_calls == [("lexical", "vacation days", 20)]  # words, unprefixed
    assert result.question == "vacation days"


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_every_retrieval_mode_answers_over_the_fake_store(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, mode: str
) -> None:
    workflow = workflow_for(fake_deps, retrieval_mode=mode)

    result = await workflow.run("How many vacation days do employees accrue?")

    assert len(result.passages) == DEFAULT_TOP_K_RERANK
    assert "vacation" in result.passages[0].text.lower()
    assert result.answer is not None


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_text_ranking_mode_over_a_store_that_ranks_no_text_is_refused_when_built(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, mode: str
) -> None:
    deps = with_ports(fake_deps, vector_store=DenseOnlyStore())

    with pytest.raises(ValueError, match=f"retrieval_mode='{mode}'.*ranks no text"):
        workflow_for(deps, retrieval_mode=mode)


def test_dense_mode_needs_no_store_that_ranks_text(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    deps = with_ports(fake_deps, vector_store=DenseOnlyStore())

    workflow_for(deps, retrieval_mode="dense")  # no error


# --- the result's timings ----------------------------------------------------------


def test_stage_timings_keys_every_stage_in_the_graph_s_order_then_the_total() -> None:
    recorded = {"rerank": 3.0, "embed": 1.0}  # in the order the stages finished

    timings = stage_timings(["embed", "rerank", "generate"], recorded, 9.5)

    assert timings == {"embed": 1.0, "rerank": 3.0, "generate": 0.0, "total": 9.5}
    assert list(timings) == ["embed", "rerank", "generate", "total"]


def test_stage_tokens_keys_each_reporting_stage_in_the_graph_s_order() -> None:
    late, early = {"prompt": 9, "completion": 2}, {"prompt": 5, "completion": 6}
    recorded = {"generate": late, "rewrite": early}  # in no particular order

    tokens = stage_tokens(["rewrite", "embed", "generate"], recorded)

    assert list(tokens.items()) == [("rewrite", early), ("generate", late)]


# --- the client's metadata filter ---------------------------------------------------

ON_PAY = MetadataCondition(attribute="topic", comparator="eq", value="pay")
ON_LEAVE = MetadataCondition(attribute="topic", comparator="ne", value="leave")


async def test_a_condition_the_client_repeats_is_in_the_result_once(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    workflow = workflow_for(fake_deps)

    result = await workflow.run(
        VACATION_QUESTION, metadata_filter=[ON_PAY, ON_LEAVE, ON_PAY]
    )

    assert result.metadata_filter == [ON_PAY, ON_LEAVE]
