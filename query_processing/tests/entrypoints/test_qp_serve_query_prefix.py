"""The query prefix, seen from outside the workflow app: what the model receives.

The embedder puts its model's query prefix in front of a question and never in
front of a passage. Each test starts the workflow app from settings, so the fake
store is seeded through the embedder under test, asks ``/retrieve`` once, and
reads the texts that reached the model or the endpoint.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.corpus import chunk, synthetic_corpus
from query_processing.entrypoints.qp_model_server import create_model_server
from query_processing.entrypoints.qp_serve import create_app
from sentence_transformers_fakes import FakeSentenceTransformersModule

SettingsFactory = Callable[..., WorkflowSettings]
QUESTION = "How many paid vacation days do employees accrue each year?"
EMBED_URL = "http://embedder.test/v3/embeddings"
FAKE_REST = {
    "components__reranker__backend": "fake",
    "components__llm__backend": "fake",
    "components__vector_store__backend": "fake",
}
# By id, with its threads pinned: no OpenVINO export and no calibration sample.
IN_PROCESS = {
    "components__embedder__backend": "local",
    "local_models__runtime": "torch",
    "components__embedder__local__threads": 1,
}


class FakeEmbeddingsEndpoint:
    """An endpoint in the openai embeddings format that records each request's texts."""

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        self.inputs.append(texts)
        rows = [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(texts))]
        return httpx.Response(200, json={"object": "list", "data": rows})


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def seeded_then_asked(
    composition: CompositionRoot, received: list[list[str]]
) -> tuple[list[str], list[list[str]]]:
    """Start the workflow app and ask ``/retrieve`` once.

    Returns the texts ``received`` at start-up, when the fake store is seeded,
    then the batches it received for the question.
    """
    with TestClient(create_app(composition)) as client:
        at_start = len(received)
        response = client.post("/retrieve", json={"question": QUESTION})
    assert response.status_code == 200
    seeded = [text for batch in received[:at_start] for text in batch]
    return seeded, received[at_start:]


def corpus_passages(settings: WorkflowSettings) -> list[str]:
    """The passages the fake store is seeded with, as their texts."""
    seed_documents = settings.components["vector_store"].fake.seed_documents
    documents = synthetic_corpus(seed_documents)
    return [passage.text for document in documents for passage in chunk(document)]


@pytest.mark.parametrize("prefix", ["query: ", ""])
def test_an_endpoint_gets_the_prefixed_question_and_the_passages_as_they_are(
    settings_factory: SettingsFactory, prefix: str
) -> None:
    endpoint = FakeEmbeddingsEndpoint()
    settings = settings_factory(
        **FAKE_REST,
        components__embedder__backend="api",
        components__embedder__api__url=EMBED_URL,
        components__embedder__query_prefix=prefix,
    )
    composition = CompositionRoot(settings, transport=httpx.MockTransport(endpoint))

    seeded, asked = seeded_then_asked(composition, endpoint.inputs)

    assert asked == [[prefix + QUESTION]]
    assert seeded == corpus_passages(settings)


def test_an_in_process_model_gets_the_prefixed_question_and_the_passages_as_they_are(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(**FAKE_REST, **IN_PROCESS)

    seeded, asked = seeded_then_asked(CompositionRoot(settings), fake_weights.encoded)

    assert asked == [["query: " + QUESTION]]
    assert seeded == corpus_passages(settings)


def test_a_model_server_embeds_what_it_receives_so_a_query_is_prefixed_once(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    # split-all's embed path: the workflow app's endpoint is our model server,
    # which serves the in-process model.
    model_server = create_model_server(
        "embedder", CompositionRoot(settings_factory(**IN_PROCESS))
    )
    settings = settings_factory(
        **FAKE_REST,
        components__embedder__backend="api",
        components__embedder__api__url=EMBED_URL,
    )
    to_the_server = httpx.ASGITransport(app=model_server)

    with TestClient(model_server):  # runs its lifespan: the model loads
        seeded, asked = seeded_then_asked(
            CompositionRoot(settings, transport=to_the_server), fake_weights.encoded
        )

    assert asked == [["query: " + QUESTION]]
    assert seeded == corpus_passages(settings)


def test_the_fake_embedder_has_no_prefix(settings_factory: SettingsFactory) -> None:
    answers = []
    for prefix in ("query: ", ""):
        settings = settings_factory(
            **FAKE_REST,
            components__embedder__backend="fake",
            components__embedder__query_prefix=prefix,
        )
        with TestClient(create_app(CompositionRoot(settings))) as client:
            answers.append(client.post("/retrieve", json={"question": QUESTION}))

    with_prefix, without = (answer.json()["passages"] for answer in answers)
    assert with_prefix == without  # retrieval scores included: the same vector
