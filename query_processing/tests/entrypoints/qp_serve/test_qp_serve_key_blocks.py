"""The ``key`` block of an ``api`` endpoint, seen from ``qp-serve`` and its app.

An endpoint names where its API key comes from: ``{source: env, variable: X}``
reads ``X`` from the environment, then from the ``.env`` file the settings read;
``{source: domino}`` sends the run token; no block sends no key. Endpoints that
name the same block share one source. An ``env`` key that is refused runs out
its retries, and the error names the variable and the restart it needs; an
``env`` variable unset or empty stops the start-up in one line. Each case runs
``qp_serve.main`` in an empty directory, its calls going to one HTTP transport
that answers in place of every endpoint and of Domino's token URL.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from uvicorn.lifespan.on import LifespanOn

from model_api_fakes import standard_answer
from query_processing.adapters.key_source.base_key_source import (
    KEY_SOURCES,
    BaseKeySource,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import MissingKeyError
from query_processing.entrypoints import qp_serve
from rag_components import shell_settings
from rag_graphs import fixture_components, fixture_graph, write_workflow_file

QUESTION = "How many paid vacation days do employees accrue each year?"
VENDOR_EMBED_URL = "https://embed.vendor.test/v1/embeddings"
VENDOR_RERANK_URL = "https://rerank.vendor.test/v2/rerank"
TOKEN_URL = "http://domino.test/access-token"
KEY_VARIABLE = "VENDOR_RERANK_KEY"
VENDOR_KEY = "co-secret-key-4711"
# Every component on its fake but the reranker, an endpoint we do not run; two
# retries, no wait between them.
VENDOR_RERANKER = {
    "QP_COMPONENTS__EMBEDDER__BACKEND": "fake",
    "QP_COMPONENTS__LLM__BACKEND": "fake",
    "QP_COMPONENTS__VECTOR_STORE__BACKEND": "fake",
    "QP_COMPONENTS__RERANKER__BACKEND": "api",
    "QP_COMPONENTS__RERANKER__API__URL": VENDOR_RERANK_URL,
    "QP_COMPONENTS__RERANKER__API__API_FORMAT": "cohere",
    "QP_COMPONENTS__RERANKER__API__MODEL": "rerank-v3.5",
    "QP_DOMINO__TOKEN_URL": TOKEN_URL,
    "QP_HTTP__RETRIES": "2",
    "QP_HTTP__BACKOFF_S": "0",
}
# The embedder on an endpoint we do not run too, for two keyed endpoints.
VENDOR_EMBEDDER = {
    "QP_COMPONENTS__EMBEDDER__BACKEND": "api",
    "QP_COMPONENTS__EMBEDDER__API__URL": VENDOR_EMBED_URL,
    "QP_COMPONENTS__EMBEDDER__API__MODEL": "embed-v1",
}
ENV_BLOCK = {"source": "env", "variable": KEY_VARIABLE}


class KeyedEndpoints:
    """Every endpoint and Domino's token URL; refuses the keys it is told to.

    It numbers the run tokens it serves and records each call's URL and
    ``Authorization``; a refused call gets a 401 that quotes its key back.
    """

    def __init__(self, *, refused: tuple[str, ...] = ()) -> None:
        self.refused = set(refused)
        self.json_escaped = (
            False  # a refusal quotes its key as JSON writing '/' as '\\/'
        )
        self.tokens_served = 0
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.tokens_served += 1
            return httpx.Response(200, text=f"run-token-{self.tokens_served}")
        auth = request.headers.get("Authorization")
        self.calls.append((str(request.url), auth))
        if auth in self.refused and self.json_escaped:
            quoted = json.dumps({"refused": auth}).replace("/", "\\/")
            return httpx.Response(401, text=quoted)
        if auth in self.refused:
            return httpx.Response(401, text=f"key refused: {auth}")
        return httpx.Response(200, json=standard_answer(json.loads(request.content)))

    def sent_to(self, url: str) -> list[str | None]:
        """The ``Authorization`` of every call made to ``url``, in order."""
        return [auth for called, auth in self.calls if called == url]


class CountingKeySource(BaseKeySource):
    """A key source a test registers: it gives ``counted-<label>`` and counts itself."""

    reads: ClassVar[tuple[str, ...]] = ("label",)
    built: ClassVar[list[CountingKeySource]] = []

    def __init__(self, label: str) -> None:
        self._label = label
        CountingKeySource.built.append(self)

    def key(self, *, may_ask: bool) -> str:
        del may_ask
        return f"counted-{self._label}"

    def refused(self, key: str) -> None:
        del key

    def described(self) -> str:
        return f"the counting source {self._label!r}"


@pytest.fixture
def endpoints() -> KeyedEndpoints:
    return KeyedEndpoints()


@pytest.fixture
def qp_serve_apps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoints: KeyedEndpoints
) -> list[FastAPI]:
    """The apps ``qp_serve.main`` would serve, in an empty directory, over ``endpoints``."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.delenv(KEY_VARIABLE, raising=False)
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


def set_environment(
    monkeypatch: pytest.MonkeyPatch, variables: Mapping[str, str]
) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


def key_variables(component: str, block: Mapping[str, str]) -> dict[str, str]:
    """The variables that set ``component``'s ``key`` block, one per key."""
    prefix = f"QP_COMPONENTS__{component.upper()}__API__KEY__"
    return {prefix + key.upper(): value for key, value in block.items()}


def served_once(apps: list[FastAPI]) -> httpx.Response:
    """Run ``qp-serve``, start the app it would serve and ask ``/retrieve`` once."""
    assert qp_serve.main([]) == 0
    with TestClient(apps[0]) as client:
        return client.post("/retrieve", json={"question": QUESTION})


# --- each block sends its key ------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "authorization"),
    [
        ({}, None),
        (ENV_BLOCK, f"Bearer {VENDOR_KEY}"),
        ({"source": "domino"}, "Bearer run-token-1"),
    ],
    ids=["no-block", "env", "domino"],
)
def test_each_key_block_sends_its_key_off_domino(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    block: dict[str, str],
    authorization: str | None,
) -> None:
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", block))
    monkeypatch.setenv(KEY_VARIABLE, VENDOR_KEY)

    assert served_once(qp_serve_apps).status_code == 200

    assert endpoints.sent_to(VENDOR_RERANK_URL) == [authorization]


@pytest.mark.parametrize("way", ["workflow-file", "variables"])
def test_a_key_block_is_set_in_the_workflow_file_or_by_its_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    way: str,
) -> None:
    set_environment(monkeypatch, VENDOR_RERANKER)
    monkeypatch.setenv(KEY_VARIABLE, VENDOR_KEY)
    if way == "variables":
        set_environment(monkeypatch, key_variables("reranker", ENV_BLOCK))
    else:
        components = fixture_components()
        components["reranker"]["api"] = {"key": ENV_BLOCK}
        workflow = write_workflow_file(tmp_path / "w.yaml", fixture_graph(), components)
        monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    assert served_once(qp_serve_apps).status_code == 200

    assert endpoints.sent_to(VENDOR_RERANK_URL) == [f"Bearer {VENDOR_KEY}"]


@pytest.mark.parametrize(
    ("in_environment", "sent"),
    [(None, "from-dotenv"), ("from-environment", "from-environment")],
    ids=["dotenv-alone", "environment-first"],
)
def test_an_env_key_is_read_from_the_environment_then_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    in_environment: str | None,
    sent: str,
) -> None:
    # The settings' own .env: a key beside the settings' variables.
    (tmp_path / ".env").write_text(
        "QP_RUN_LABEL=from-dotenv\n"
        f"{KEY_VARIABLE}=from-dotenv\n"
        "SOMETHING_ELSE=unrelated\n",
        encoding="utf-8",
    )
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", ENV_BLOCK))
    if in_environment is not None:
        monkeypatch.setenv(KEY_VARIABLE, in_environment)

    response = served_once(qp_serve_apps)

    assert response.status_code == 200
    assert response.headers["X-RAG-Run-Label"] == "from-dotenv"
    assert endpoints.sent_to(VENDOR_RERANK_URL) == [f"Bearer {sent}"]


@pytest.mark.parametrize(
    ("env_file", "sent"),
    [(None, None), ("other.env", "from-other"), (".env", "from-dotenv")],
    ids=["none", "another", "the-default"],
)
def test_an_env_key_is_read_from_the_dotenv_file_the_settings_were_built_with(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    env_file: str | None,
    sent: str | None,
) -> None:
    (tmp_path / ".env").write_text(f"{KEY_VARIABLE}=from-dotenv\n", encoding="utf-8")
    other = f"{KEY_VARIABLE}=from-other\n"
    (tmp_path / "other.env").write_text(other, encoding="utf-8")
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", ENV_BLOCK))
    settings = WorkflowSettings(_env_file=env_file)
    app = qp_serve.create_app(
        CompositionRoot(settings, transport=httpx.MockTransport(endpoints))
    )

    if sent is None:
        with pytest.raises(MissingKeyError, match=KEY_VARIABLE), TestClient(app):
            pass
        return
    with TestClient(app) as client:
        assert client.post("/retrieve", json={"question": QUESTION}).status_code == 200
    assert endpoints.sent_to(VENDOR_RERANK_URL) == [f"Bearer {sent}"]


# --- one source a block, and the source the settings name ------------------------------


def test_the_source_the_settings_name_is_the_one_that_runs(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
) -> None:
    # Registered as a key source is, under a name the settings then name: the
    # start-up check of registered names accepts it, and its key is sent.
    monkeypatch.setattr(KEY_SOURCES, "_entries", dict(KEY_SOURCES._entries))
    KEY_SOURCES.register("counting")(CountingKeySource)
    monkeypatch.setattr(CountingKeySource, "built", [])
    block = {"source": "counting", "label": "mine"}
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", block))

    assert served_once(qp_serve_apps).status_code == 200

    assert endpoints.sent_to(VENDOR_RERANK_URL) == ["Bearer counted-mine"]
    assert len(CountingKeySource.built) == 1


@pytest.mark.parametrize(("embedder_label", "built"), [("shared", 1), ("its-own", 2)])
def test_components_share_a_source_when_they_name_the_same_block(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    embedder_label: str,
    built: int,
) -> None:
    # As two env blocks that name the same variable: one block, one source.
    monkeypatch.setattr(KEY_SOURCES, "_entries", dict(KEY_SOURCES._entries))
    KEY_SOURCES.register("counting")(CountingKeySource)
    monkeypatch.setattr(CountingKeySource, "built", [])
    keyed = key_variables("reranker", {"source": "counting", "label": "shared"})
    keyed |= key_variables("embedder", {"source": "counting", "label": embedder_label})
    set_environment(monkeypatch, VENDOR_RERANKER | VENDOR_EMBEDDER | keyed)

    assert served_once(qp_serve_apps).status_code == 200

    embedded = set(endpoints.sent_to(VENDOR_EMBED_URL))
    assert embedded == {f"Bearer counted-{embedder_label}"}
    assert len(CountingKeySource.built) == built


def test_two_api_components_that_name_domino_share_one_run_token(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
) -> None:
    domino = {"source": "domino"}
    keyed = key_variables("reranker", domino) | key_variables("embedder", domino)
    set_environment(monkeypatch, VENDOR_RERANKER | VENDOR_EMBEDDER | keyed)

    assert served_once(qp_serve_apps).status_code == 200

    assert endpoints.tokens_served == 1
    assert set(endpoints.sent_to(VENDOR_EMBED_URL)) == {"Bearer run-token-1"}
    assert endpoints.sent_to(VENDOR_RERANK_URL) == ["Bearer run-token-1"]


# --- a refused env key, and an env variable not set -------------------------------------


def test_a_refused_env_key_runs_out_its_retries_and_names_its_variable(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
) -> None:
    endpoints.refused.add(f"Bearer {VENDOR_KEY}")
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", ENV_BLOCK))
    monkeypatch.setenv(KEY_VARIABLE, VENDOR_KEY)

    response = served_once(qp_serve_apps)

    assert response.status_code == 503
    body = response.json()
    assert (body["error"], body["target"]) == (
        "reranker_unavailable",
        VENDOR_RERANK_URL,
    )
    assert "HTTP 401" in body["detail"]
    assert KEY_VARIABLE in body["detail"]
    assert "restart the App" in body["detail"]
    assert VENDOR_KEY not in response.text
    assert endpoints.sent_to(VENDOR_RERANK_URL) == [f"Bearer {VENDOR_KEY}"] * 3
    assert endpoints.tokens_served == 0


def test_a_key_echoed_json_escaped_is_masked_and_the_log_line_says_restart(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A base64 key holds '/', which many JSON encoders write as '\/'.
    key = f'{VENDOR_KEY}/x"y'
    endpoints.refused.add(f"Bearer {key}")
    endpoints.json_escaped = True
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", ENV_BLOCK))
    monkeypatch.setenv(KEY_VARIABLE, key)

    with caplog.at_level(logging.INFO, logger="query_processing"):
        response = served_once(qp_serve_apps)

    assert response.status_code == 503
    assert VENDOR_KEY not in response.text
    messages = [record.getMessage() for record in caplog.records]
    logged = [json.loads(m) for m in messages if '"rag_request"' in m]
    [failed] = [entry for entry in logged if entry["status"] == 503]
    assert "restart the App" in failed["detail"]
    assert VENDOR_KEY not in caplog.text


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_an_env_variable_unset_or_empty_stops_the_start_up_in_one_line(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    endpoints: KeyedEndpoints,
    value: str | None,
) -> None:
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", ENV_BLOCK))
    if value is not None:
        monkeypatch.setenv(KEY_VARIABLE, value)

    assert qp_serve.main([]) == 0  # reading the settings needs no key
    with pytest.raises(MissingKeyError) as refused, TestClient(qp_serve_apps[0]):
        pass

    [line] = str(refused.value).splitlines()
    assert KEY_VARIABLE in line
    assert "'reranker'" in line
    assert endpoints.calls == []


async def test_an_env_variable_not_set_reaches_uvicorn_as_one_line(
    monkeypatch: pytest.MonkeyPatch,
    qp_serve_apps: list[FastAPI],
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_environment(monkeypatch, VENDOR_RERANKER | key_variables("reranker", ENV_BLOCK))
    assert qp_serve.main([]) == 0
    config = uvicorn.Config(qp_serve_apps[0], lifespan="on", log_config=None)
    config.load()
    lifespan = LifespanOn(config)

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await lifespan.startup()

    assert lifespan.should_exit
    messages = [record.getMessage() for record in caplog.records]
    [message] = [m for m in messages if m.startswith("qp-serve: cannot start: ")]
    assert KEY_VARIABLE in message
    assert "'reranker'" in message
    assert "\n" not in message
    assert not any("Traceback" in m for m in messages)
