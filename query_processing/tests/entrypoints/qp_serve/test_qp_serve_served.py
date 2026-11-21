"""The ``served`` and ``api`` backends, seen from outside the workflow app.

A component on ``served`` is a model server of ours: it is called at the URL it
is given, in the standard format of its type, under its alias, with the Domino
run token when ``qp-serve --domino`` says the app runs on Domino and with no
key elsewhere. A component on ``api`` is an endpoint we do not run, called with
its own URL, API format, key block and model name wherever the app runs. The
endpoints and Domino's token URL are one HTTP transport that answers in their
place and records what each call carried.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_api_fakes import standard_answer
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints import qp_serve
from query_processing.entrypoints.qp_serve import create_app
from rag_components import shell_settings

SettingsFactory = Callable[..., WorkflowSettings]
QUESTION = "How many paid vacation days do employees accrue each year?"
EMBED_URL = "http://embedder-app.test/v3/embeddings"
RERANK_URL = "http://reranker-app.test/v3/rerank"
CHAT_URL = "http://llm-app.test/v3/chat/completions"
VENDOR_RERANK_URL = "https://rerank.vendor.test/v2/rerank"
TOKEN_URL = "http://domino.test/access-token"
RUN_TOKEN = "run-token"
EMBEDDER_ALIAS = "fixture-embedder"
RERANKER_ALIAS = "fixture-reranker"
# One served reranker; every other component on its fake, so the run token is
# the only Authorization a call can carry.
SERVED_RERANKER = {
    "QP_COMPONENTS__EMBEDDER__BACKEND": "fake",
    "QP_COMPONENTS__RERANKER__BACKEND": "served",
    "QP_COMPONENTS__RERANKER__SERVED__URL": RERANK_URL,
    "QP_COMPONENTS__LLM__BACKEND": "fake",
    "QP_COMPONENTS__VECTOR_STORE__BACKEND": "fake",
    "QP_DOMINO__TOKEN_URL": TOKEN_URL,
}


class FakeModelEndpoints:
    """Every endpoint and Domino's token URL, answering each in its standard format.

    An embedding request is told apart by its ``input``, a rerank request by its
    ``documents``, a chat request by its ``messages``; each call is recorded with
    its URL, body and ``Authorization`` header.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self.tokens_served = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.tokens_served += 1
            return httpx.Response(200, text=RUN_TOKEN)
        body = json.loads(request.content)
        auth = request.headers.get("Authorization")
        self.calls.append((str(request.url), body, auth))
        return httpx.Response(200, json=standard_answer(body))

    def sent_to(self, url: str) -> list[tuple[dict[str, Any], str | None]]:
        """The body and ``Authorization`` of every call made to ``url``."""
        return [(body, auth) for called, body, auth in self.calls if called == url]


@pytest.fixture
def endpoints() -> FakeModelEndpoints:
    return FakeModelEndpoints()


@pytest.fixture
def qp_serve_apps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoints: FakeModelEndpoints
) -> list[FastAPI]:
    """The apps ``qp_serve.main`` would serve, their calls going to ``endpoints``.

    No ``QP_*`` or ``DOMINO_*`` variable of the shell and no ``.env`` is read:
    a test sets what it needs.
    """
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    transport = httpx.MockTransport(endpoints)

    def composition_over_endpoints(
        settings: WorkflowSettings, **options: Any
    ) -> CompositionRoot:
        return CompositionRoot(settings, transport=transport, **options)

    apps: list[FastAPI] = []
    monkeypatch.setattr(qp_serve, "CompositionRoot", composition_over_endpoints)
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda app, **kw: apps.append(app))
    return apps


def retrieve(app: FastAPI) -> dict[str, Any]:
    """Start ``app`` and ask ``/retrieve`` once."""
    with TestClient(app) as client:
        response = client.post("/retrieve", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response.json()


def set_environment(monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


# --- a served component: its URL, the standard format, its alias ---------------------


def test_served_components_speak_the_standard_format_at_their_url_under_their_alias(
    settings_factory: SettingsFactory, endpoints: FakeModelEndpoints
) -> None:
    settings = settings_factory(
        components__embedder__backend="served",
        components__embedder__served__url=EMBED_URL,
        components__reranker__backend="served",
        components__reranker__served__url=RERANK_URL,
        components__llm__backend="served",
        components__llm__served__url=CHAT_URL,
        components__llm__alias="llama-3",
        components__vector_store__backend="fake",
        graph__stages__rerank__top_k=3,
    )
    composition = CompositionRoot(settings, transport=httpx.MockTransport(endpoints))

    with TestClient(create_app(composition)) as client:
        response = client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "Twenty-five days."
    # OpenAI's embeddings format, the query with the embedder's prefix.
    embedded = endpoints.sent_to(EMBED_URL)
    query = {"model": EMBEDDER_ALIAS, "input": [f"query: {QUESTION}"]}
    assert (query, None) in embedded
    assert {body["model"] for body, _ in embedded} == {EMBEDDER_ALIAS}
    # Cohere's rerank format, the candidates of the retrieve stage.
    [(reranked, _)] = endpoints.sent_to(RERANK_URL)
    assert (reranked["model"], reranked["query"], reranked["top_n"]) == (
        RERANKER_ALIAS,
        QUESTION,
        3,
    )
    assert len(reranked["documents"]) == 20
    # OpenAI's chat format, under the LLM's alias.
    [(chat, _)] = endpoints.sent_to(CHAT_URL)
    assert (chat["model"], chat["temperature"]) == ("llama-3", 0.0)
    assert [message["role"] for message in chat["messages"]] == ["system", "user"]
    # Off Domino, a model server of ours is called with no key.
    assert {auth for _, _, auth in endpoints.calls} == {None}
    assert endpoints.tokens_served == 0


def test_the_served_blocks_take_what_each_request_carries(
    settings_factory: SettingsFactory, endpoints: FakeModelEndpoints
) -> None:
    settings = settings_factory(
        components__embedder__backend="served",
        components__embedder__served__url=EMBED_URL,
        components__embedder__served__batch_size=50,
        components__reranker__backend="fake",
        components__llm__backend="served",
        components__llm__served__url=CHAT_URL,
        components__llm__served__temperature=0.7,
        components__llm__alias="llama-3",
        components__vector_store__backend="fake",
    )
    composition = CompositionRoot(settings, transport=httpx.MockTransport(endpoints))

    with TestClient(create_app(composition)) as client:
        assert client.post("/query", json={"question": QUESTION}).status_code == 200

    batches = [len(body["input"]) for body, _ in endpoints.sent_to(EMBED_URL)]
    assert max(batches) == 50  # the store is seeded 50 records per request
    [(chat, _)] = endpoints.sent_to(CHAT_URL)
    assert chat["temperature"] == 0.7


# --- on Domino and off it --------------------------------------------------------------


def test_on_domino_a_served_component_is_called_with_the_run_token(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: FakeModelEndpoints,
) -> None:
    set_environment(monkeypatch, SERVED_RERANKER)

    assert qp_serve.main(["--domino"]) == 0
    retrieve(qp_serve_apps[0])

    assert endpoints.tokens_served == 1  # fetched once, then reused
    [(body, auth)] = endpoints.sent_to(RERANK_URL)
    assert (body["model"], auth) == (RERANKER_ALIAS, f"Bearer {RUN_TOKEN}")


def test_on_domino_a_served_llm_is_called_with_the_run_token_under_its_alias(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: FakeModelEndpoints,
) -> None:
    # What qp-deploy hands the workflow App for an LLM in an OVMS App of its own.
    set_environment(
        monkeypatch,
        {
            **SERVED_RERANKER,
            "QP_COMPONENTS__RERANKER__BACKEND": "fake",
            "QP_COMPONENTS__LLM__BACKEND": "served",
            "QP_COMPONENTS__LLM__SERVED__URL": CHAT_URL,
            "QP_COMPONENTS__LLM__ALIAS": "llama-3",
        },
    )

    assert qp_serve.main(["--domino"]) == 0
    with TestClient(qp_serve_apps[0]) as client:
        response = client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    [(body, auth)] = endpoints.sent_to(CHAT_URL)
    assert (body["model"], auth) == ("llama-3", f"Bearer {RUN_TOKEN}")


def test_off_domino_a_served_component_is_called_with_no_key(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: FakeModelEndpoints,
) -> None:
    set_environment(monkeypatch, SERVED_RERANKER)

    assert qp_serve.main([]) == 0
    retrieve(qp_serve_apps[0])

    assert endpoints.tokens_served == 0
    [(body, auth)] = endpoints.sent_to(RERANK_URL)
    assert (body["model"], auth) == (RERANKER_ALIAS, None)


def test_a_split_run_on_a_laptop_moves_one_component_by_its_backend_and_url(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: FakeModelEndpoints,
) -> None:
    # The README's laptop split: the reranker behind a model server started by
    # hand, by its backend and URL alone. No topology variable is involved.
    set_environment(monkeypatch, SERVED_RERANKER)

    assert qp_serve.main([]) == 0
    with TestClient(qp_serve_apps[0]) as client:
        response = client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    [(body, _)] = endpoints.sent_to(RERANK_URL)
    assert body["query"] == QUESTION
    assert {url for url, _, _ in endpoints.calls} == {RERANK_URL}
    # No topology alias was given: none is reported, and the run label names
    # where each component ran.
    assert "X-RAG-Topology" not in response.headers
    assert response.json()["topology"] is None
    label = "embedder=fake+reranker=served+llm=fake+vector_store=fake"
    assert response.headers["X-RAG-Run-Label"] == label


@pytest.mark.parametrize(
    ("key", "authorization"),
    [({}, None), ({"SOURCE": "env", "VARIABLE": "VENDOR_KEY"}, "Bearer co-key")],
    ids=["no-key", "its-key"],
)
def test_on_domino_an_api_component_keeps_its_own_url_format_key_and_model(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: FakeModelEndpoints,
    key: dict[str, str],
    authorization: str | None,
) -> None:
    # The Domino flag no longer gives an api component the run token: an
    # endpoint we do not run is called as its block says.
    vendor = {
        "QP_COMPONENTS__RERANKER__BACKEND": "api",
        "QP_COMPONENTS__RERANKER__API__URL": VENDOR_RERANK_URL,
        "QP_COMPONENTS__RERANKER__API__API_FORMAT": "cohere",
        "QP_COMPONENTS__RERANKER__API__MODEL": "rerank-v3.5",
        "VENDOR_KEY": "co-key",
    }
    block = {f"QP_COMPONENTS__RERANKER__API__KEY__{k}": v for k, v in key.items()}
    set_environment(monkeypatch, SERVED_RERANKER | vendor | block)

    assert qp_serve.main(["--domino"]) == 0
    retrieve(qp_serve_apps[0])

    assert endpoints.sent_to(RERANK_URL) == []  # the served block is not read
    [(body, auth)] = endpoints.sent_to(VENDOR_RERANK_URL)
    assert (body["model"], auth, endpoints.tokens_served) == (
        "rerank-v3.5",
        authorization,
        0,
    )


# --- no URL: refused at start-up ---------------------------------------------------------


@pytest.mark.parametrize("backend", ["served", "api"])
@pytest.mark.parametrize("component", ["embedder", "reranker", "llm"])
def test_a_component_called_over_http_without_a_url_stops_qp_serve_naming_its_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    qp_serve_apps: list[FastAPI],
    backend: str,
    component: str,
) -> None:
    upper = component.upper()
    no_url = {
        f"QP_COMPONENTS__{upper}__BACKEND": backend,
        f"QP_COMPONENTS__{upper}__{backend.upper()}__URL": "",
        "QP_COMPONENTS__LLM__ALIAS": "llama-3",
    }
    set_environment(monkeypatch, SERVED_RERANKER | no_url)

    with pytest.raises(SystemExit) as exited:
        qp_serve.main(["--domino"])

    assert exited.value.code == 2
    needed = f"needs QP_COMPONENTS__{upper}__{backend.upper()}__URL, the full route"
    assert needed in capsys.readouterr().err
    assert qp_serve_apps == []


def test_a_served_llm_without_an_alias_stops_qp_serve_naming_its_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    qp_serve_apps: list[FastAPI],
) -> None:
    # A model server serves a model under its alias; the LLM has no source id.
    served_llm = {
        "QP_COMPONENTS__LLM__BACKEND": "served",
        "QP_COMPONENTS__LLM__SERVED__URL": CHAT_URL,
    }
    set_environment(monkeypatch, SERVED_RERANKER | served_llm)

    with pytest.raises(SystemExit) as exited:
        qp_serve.main([])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "QP_COMPONENTS__LLM__BACKEND='served' needs QP_COMPONENTS__LLM__ALIAS" in err
    assert qp_serve_apps == []
