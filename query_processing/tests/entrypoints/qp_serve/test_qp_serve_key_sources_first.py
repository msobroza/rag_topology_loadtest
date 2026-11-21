"""The key sources of the components the app calls open before any component opens.

So an ``env`` variable unset is refused before a local model loads, and a key
source that cannot give its key when its component opens ends the start-up in
one line. The embedder opens on a backend the test registers, which records
that it opened: it stands for a local model that takes long to load.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from uvicorn.lifespan.on import LifespanOn

from query_processing.adapters.embedder.base_embedder import BaseEmbedder
from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.composition.composition_root import CompositionRoot
from query_processing.composition.openers.component_openers import EMBEDDER_OPENERS
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import MissingKeyError
from query_processing.entrypoints import qp_serve
from rag_components import shell_settings
from rag_graphs import fixture_components, fixture_graph, write_workflow_file

UNSET = "KEY_NOBODY_SET"
TOKEN_URL = "http://domino.test/access-token"
UNSET_BLOCK = {"source": "env", "variable": UNSET}
RECORDED_EMBEDDER = {"QP_COMPONENTS__EMBEDDER__BACKEND": "recorded"}
ON_FAKES = {
    f"QP_COMPONENTS__{name}__BACKEND": "fake"
    for name in ("RERANKER", "LLM", "VECTOR_STORE")
}
API_RERANKER = {
    "QP_COMPONENTS__RERANKER__BACKEND": "api",
    "QP_COMPONENTS__RERANKER__API__URL": "https://rerank.vendor.test/v2/rerank",
    "QP_COMPONENTS__RERANKER__API__API_FORMAT": "cohere",
    "QP_COMPONENTS__RERANKER__API__KEY__SOURCE": "env",
    "QP_COMPONENTS__RERANKER__API__KEY__VARIABLE": UNSET,
}
ES_STORE = {
    "QP_COMPONENTS__VECTOR_STORE__BACKEND": "elasticsearch",
    "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL": "http://127.0.0.1:1",
}
ES_KEY = "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__KEY"


@pytest.fixture(autouse=True)
def recorded_opens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """No QP_* of the shell, an empty directory, and the embedder's ``recorded`` backend."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.delenv(UNSET, raising=False)
    monkeypatch.chdir(tmp_path)
    opened: list[str] = []

    async def open_recorded(component: Any, shared: Any) -> BaseEmbedder:
        opened.append("embedder")
        return FakeEmbedder()

    monkeypatch.setattr(EMBEDDER_OPENERS, "_entries", dict(EMBEDDER_OPENERS._entries))
    EMBEDDER_OPENERS.register("recorded")(open_recorded)
    return opened


def app_on(variables: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Any:
    for name, value in (ON_FAKES | RECORDED_EMBEDDER | variables).items():
        monkeypatch.setenv(name, value)
    settings = WorkflowSettings(_env_file=None)
    transport = httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
    return qp_serve.create_app(CompositionRoot(settings, transport=transport))


@pytest.mark.parametrize(
    "variables",
    [
        API_RERANKER,
        ES_STORE | {f"{ES_KEY}__SOURCE": "env", f"{ES_KEY}__VARIABLE": UNSET},
    ],
    ids=["an-api-reranker", "the-store"],
)
def test_an_unset_variable_is_refused_before_any_component_opens(
    monkeypatch: pytest.MonkeyPatch,
    recorded_opens: list[str],
    variables: dict[str, str],
) -> None:
    app = app_on(variables, monkeypatch)

    with pytest.raises(MissingKeyError, match=UNSET), TestClient(app):
        pass

    assert recorded_opens == []


def test_the_key_source_of_a_component_the_app_does_not_call_is_not_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recorded_opens: list[str]
) -> None:
    components = fixture_components()
    spare = {"backend": "api", "type": "llm", "api": {"url": "https://llm.test/v1"}}
    components["spare_llm"] = spare | {"api": spare["api"] | {"key": UNSET_BLOCK}}
    workflow = write_workflow_file(tmp_path / "w.yaml", fixture_graph(), components)
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    with TestClient(app_on({}, monkeypatch)) as client:
        assert client.get("/healthz").status_code == 200

    assert recorded_opens == ["embedder"]


async def test_a_domino_store_key_that_cannot_be_had_ends_the_start_up_in_one_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The token URL answers 503: the store asks for its key when it opens.
    domino = {f"{ES_KEY}__SOURCE": "domino", "QP_DOMINO__TOKEN_URL": TOKEN_URL}
    app = app_on(ES_STORE | domino | {"QP_HTTP__RETRIES": "0"}, monkeypatch)
    config = uvicorn.Config(app, lifespan="on", log_config=None)
    config.load()
    lifespan = LifespanOn(config)

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await lifespan.startup()

    assert lifespan.should_exit
    messages = [record.getMessage() for record in caplog.records]
    [message] = [m for m in messages if m.startswith("qp-serve: cannot start: ")]
    assert "'vector_store' component's API key" in message
    assert TOKEN_URL in message
    assert not any("Traceback" in m for m in messages)
