"""The three paths of the RAG's graph, told apart by what the fakes were asked.

The app is built from this directory's workflow file with every component on
its fake backend, and driven through its HTTP API. The components the paths
differ by are named fakes that record what they were asked: the two embedders,
the router and the LLM. A first message is answered as before the router
existed; a message with a conversational history is routed; and one the router
says leans on it is rewritten first, into the standalone query the rest of the
graph searches with, with conditions when the LLM names a source. ``timings_ms``
cannot tell the paths apart: it keeps a key for a stage that did not run;
``tokens`` can, since only a stage whose LLM reported its counts has a key.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.router.fake.fake_router import FakeRouter
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.routes import RouteChoice, RouteRequest
from query_processing.entrypoints.qp_serve import create_app
from query_processing.extension_api import GenerateAnswer

from workflow_on_fakes import all_fake_settings

FIRST_MESSAGE = "Que dit l'AMF sur la franchise ?"
FOLLOW_UP = "Et pour un véhicule loué ?"
ON_AMF = {"attribute": "source", "comparator": "in", "value": ["AMF"]}
EARLIER_TURNS = [
    {"role": "user", "content": FIRST_MESSAGE, "metadata_filter": [ON_AMF]},
    {"role": "assistant", "content": "Elle est de 500 $."},
]
STANDALONE = "Que dit l'AMF sur la franchise pour un véhicule loué ?"
# The router's routes as the variable of their path gives them: the fake
# router chooses the first, so their order decides the path of a follow-up.
FILE_ORDER = '["self_contained", "leans_on_history"]'
LEANING_FIRST = '["leans_on_history", "self_contained"]'
Messages = list[dict[str, str]]


class RecordingEmbedder(FakeEmbedder):
    """The fake embedder, recording each query it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


class RecordingRouter(FakeRouter):
    """The fake router on the route a test gives it, recording each vector asked."""

    def __init__(self, route: str) -> None:
        super().__init__(route)
        self.embeddings: list[list[float]] = []

    def invoke(self, request: RouteRequest) -> RouteChoice:
        self.embeddings.append(request.embedding)
        return super().invoke(request)


class RecordingLLM(FakeLLM):
    """The fake LLM, recording the messages and the schema of each call.

    Args:
        rewrite: What it answers under a schema; None for the fake's own
            instance, which echoes the message and names no condition.
    """

    def __init__(self, rewrite: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.rewrite = rewrite
        self.asked: list[tuple[Messages, type[BaseModel] | None]] = []

    async def generate(
        self,
        messages: Messages,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        self.asked.append((messages, output_schema))
        if output_schema is not None and self.rewrite is not None:
            return GenerateAnswer(text=json.dumps(self.rewrite))
        return await super().generate(messages, output_schema, max_tokens)

    def schemas(self) -> list[type[BaseModel] | None]:
        """The schema of each call, in order: None for a call that asked for text."""
        return [schema for _, schema in self.asked]


@dataclasses.dataclass
class AppOnRecordingFakes:
    """One app on recording fakes, each field the fake of the component it names."""

    router: RecordingRouter
    llm: RecordingLLM = dataclasses.field(default_factory=RecordingLLM)
    embedder: RecordingEmbedder = dataclasses.field(default_factory=RecordingEmbedder)
    router_embedder: RecordingEmbedder = dataclasses.field(
        default_factory=RecordingEmbedder
    )

    def query(self, settings: WorkflowSettings, **body: object) -> dict[str, Any]:
        """Start the app on these fakes and post ``body`` to /query; its answer."""
        ports = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        with TestClient(create_app(CompositionRoot(settings, **ports))) as app:
            response = app.post("/query", json=body)
        assert response.status_code == 200, response.text
        answered: dict[str, Any] = response.json()
        return answered


def routed_to(route: str, **llm: Any) -> AppOnRecordingFakes:
    """An app whose router says ``route`` for every message it is asked about."""
    return AppOnRecordingFakes(router=RecordingRouter(route), llm=RecordingLLM(**llm))


def test_a_first_message_skips_the_router_and_the_rewrite(
    all_fake: WorkflowSettings,
) -> None:
    app = routed_to("leans_on_history")  # which nothing asks

    answered = app.query(all_fake, question=FIRST_MESSAGE)

    assert app.router_embedder.queries == []
    assert app.router.embeddings == []
    assert app.llm.schemas() == [None]  # the answer alone
    assert app.embedder.queries == [FIRST_MESSAGE]
    assert list(answered["tokens"]) == ["generate"]  # no rewrite key


def test_a_follow_up_that_stands_alone_is_routed_and_not_rewritten(
    all_fake: WorkflowSettings,
) -> None:
    app = routed_to("self_contained")

    answered = app.query(
        all_fake, question=FOLLOW_UP, conversational_history=EARLIER_TURNS
    )

    # The router is asked about the message as the user sent it, embedded by
    # its own embedder; the store's embedder then embeds the same message.
    assert app.router_embedder.queries == [FOLLOW_UP]
    assert app.router.embeddings == [FakeEmbedder().embed_text(FOLLOW_UP)]
    assert app.llm.schemas() == [None]
    assert app.embedder.queries == [FOLLOW_UP]
    assert list(answered["tokens"]) == ["generate"]  # no rewrite key


def test_a_follow_up_the_router_says_leans_on_the_conversation_is_rewritten(
    all_fake: WorkflowSettings,
) -> None:
    app = routed_to("leans_on_history")

    answered = app.query(
        all_fake, question=FOLLOW_UP, conversational_history=EARLIER_TURNS
    )

    assert app.router_embedder.queries == [FOLLOW_UP]
    assert len(app.router.embeddings) == 1
    (rewrite, schema), (_, no_schema) = app.llm.asked  # the rewrite, the answer
    assert (schema is not None, no_schema) == (True, None)
    # The rewrite prompt: the conditions of the earlier message, the earlier
    # turns spliced, then the message to rewrite.
    assert f"restricted to: {json.dumps([ON_AMF])}" in rewrite[0]["content"]
    assert rewrite[1:] == [
        {"role": "user", "content": FIRST_MESSAGE},
        {"role": "assistant", "content": "Elle est de 500 $."},
        {"role": "user", "content": FOLLOW_UP},
    ]
    # What is searched is the rewrite, here the fake LLM's echo of the message.
    assert app.embedder.queries == [f"[fake-llm] {FOLLOW_UP}"]
    assert answered["metadata_filter"] == []  # the fake names no source
    # Both stages that called the LLM report its counts, in the graph's order.
    assert list(answered["tokens"]) == ["rewrite", "generate"]
    assert all(counts["completion"] > 0 for counts in answered["tokens"].values())


def test_the_rewrite_is_asked_for_conditions_on_the_stores_source_only(
    all_fake: WorkflowSettings,
) -> None:
    app = routed_to("leans_on_history")
    app.query(all_fake, question=FOLLOW_UP, conversational_history=EARLIER_TURNS)
    schema = app.llm.schemas()[0]
    assert schema is not None

    def rewritten(**condition: object) -> BaseModel:
        return schema.model_validate(
            {"standalone_query": STANDALONE, "metadata_filter": [condition]}
        )

    assert list(schema.model_fields) == ["standalone_query", "metadata_filter"]
    rewritten(**ON_AMF)  # a declared source: accepted
    for refused in (
        {"attribute": "source", "comparator": "eq", "value": "SAAQ"},
        {"attribute": "topic", "comparator": "eq", "value": "AMF"},
    ):
        with pytest.raises(ValidationError):
            rewritten(**refused)


def test_a_rewrites_conditions_join_the_metadata_filter_and_its_query_is_searched(
    all_fake: WorkflowSettings,
) -> None:
    rewrite = {"standalone_query": STANDALONE, "metadata_filter": [ON_AMF]}
    app = routed_to("leans_on_history", rewrite=rewrite)

    answered = app.query(
        all_fake, question=FOLLOW_UP, conversational_history=EARLIER_TURNS
    )

    assert app.embedder.queries == [STANDALONE]
    assert answered["metadata_filter"] == [ON_AMF]
    # No passage of the fake store's synthetic corpus carries a source.
    assert answered["passages"] == []
    # The answer is still asked about the message as the user sent it.
    answer_prompt, _ = app.llm.asked[-1]
    assert answer_prompt[-1]["content"].endswith(f"Question: {FOLLOW_UP}")


@pytest.mark.parametrize(
    ("routes", "rewrites"),
    [(FILE_ORDER, 0), (LEANING_FIRST, 1)],
    ids=["self-contained-first", "leaning-first"],
)
def test_the_fake_router_of_the_settings_decides_whether_a_follow_up_is_rewritten(
    use_case_dir: Path, monkeypatch: pytest.MonkeyPatch, routes: str, rewrites: int
) -> None:
    # The router here is the one the settings build, not a test's: the
    # toolkit's fake, which chooses the first of the routes it lists.
    settings = all_fake_settings(monkeypatch, QP_COMPONENTS__ROUTER__ROUTES=routes)
    llm = FakeLLM()
    body = {"question": FOLLOW_UP, "conversational_history": EARLIER_TURNS}

    with TestClient(create_app(CompositionRoot(settings, llm=llm))) as app:
        response = app.post("/retrieve", json=body)

    assert response.status_code == 200, response.text
    # /retrieve asks the LLM for no answer: its only call is the rewrite.
    assert llm.calls == rewrites
    assert len(response.json()["passages"]) == 10
