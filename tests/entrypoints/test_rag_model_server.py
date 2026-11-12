"""The FastAPI model server must be indistinguishable from OVMS for the ``api`` adapters."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from model_api_fakes import endpoint_client
from rag_load_test.adapters.fakes.fake_models import FakeEmbedder, FakeReranker
from rag_load_test.adapters.model_api.api_embedder import ApiEmbedder
from rag_load_test.adapters.model_api.api_reranker import ApiReranker
from rag_load_test.adapters.model_api.cohere_rerank_format import CohereRerankFormat
from rag_load_test.adapters.model_api.model_server_contract import UNKNOWN_MODEL_ERROR
from rag_load_test.adapters.model_api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)
from rag_load_test.adapters.rag_ports import RagPorts
from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.model_api import ApiEndpoint
from rag_load_test.domain.passages import ScoredPassage
from rag_load_test.entrypoints.rag_model_server import create_model_server, main

EMBED_ALIAS = "snowflake-arctic-embed-l-v2.0"  # RagSettings().embedding_api_model
RERANK_ALIAS = "crossencoder-camembert-base-mmarcoFR"  # RagSettings().rerank_api_model


def api_reranker(client: httpx.AsyncClient, alias: str) -> ApiReranker:
    """An ``ApiReranker`` at ``http://models/v3/rerank`` without auth, no sleeping."""
    endpoint = ApiEndpoint("http://models/v3/rerank", "cohere", "", alias)
    rerank_client = endpoint_client(client, endpoint, component="reranker")
    return ApiReranker(rerank_client, CohereRerankFormat())


def api_embedder(client: httpx.AsyncClient, alias: str) -> ApiEmbedder:
    """An ``ApiEmbedder`` at ``http://models/v3/embeddings`` without auth."""
    endpoint = ApiEndpoint("http://models/v3/embeddings", "openai", "", alias)
    embed_client = endpoint_client(client, endpoint, component="embedder")
    return ApiEmbedder(embed_client, OpenAIEmbeddingsFormat(), batch_size=64)


@pytest.fixture
def embedder_app(settings_factory: Callable[..., RagSettings]):
    ports = RagPorts(settings_factory(), embedder=FakeEmbedder(dim=4))
    with TestClient(create_model_server("embedder", ports)) as client:
        yield client


@pytest.fixture
def reranker_app(settings_factory: Callable[..., RagSettings]):
    ports = RagPorts(settings_factory(), reranker=FakeReranker())
    with TestClient(create_model_server("reranker", ports)) as client:
        yield client


def test_embeddings_endpoint_matches_ovms_shape(embedder_app: TestClient) -> None:
    resp = embedder_app.post(
        "/v3/embeddings", json={"model": EMBED_ALIAS, "input": ["a b", "c d"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == EMBED_ALIAS
    assert [row["index"] for row in body["data"]] == [0, 1]
    assert [row["object"] for row in body["data"]] == ["embedding", "embedding"]
    assert all(len(row["embedding"]) == 4 for row in body["data"])


def test_embeddings_accepts_a_single_string(embedder_app: TestClient) -> None:
    body = embedder_app.post(
        "/v3/embeddings", json={"model": EMBED_ALIAS, "input": "hi"}
    ).json()
    assert len(body["data"]) == 1


def test_rerank_endpoint_matches_ovms_shape(reranker_app: TestClient) -> None:
    payload = {
        "model": RERANK_ALIAS,
        "query": "vacation days",
        "documents": ["cafeteria lunch", "vacation days policy", "unrelated"],
        "top_n": 2,
    }
    body = reranker_app.post("/v3/rerank", json=payload).json()
    assert [r["index"] for r in body["results"]] == [1, 0]
    scores = [r["relevance_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_without_top_n_returns_every_document(reranker_app: TestClient) -> None:
    payload = {"model": RERANK_ALIAS, "query": "q", "documents": ["a", "b", "c"]}
    assert len(reranker_app.post("/v3/rerank", json=payload).json()["results"]) == 3


def test_embeddings_for_an_unknown_model_is_404_like_ovms(
    embedder_app: TestClient,
) -> None:
    resp = embedder_app.post("/v3/embeddings", json={"model": "m", "input": "hi"})
    assert resp.status_code == 404
    assert resp.json() == {"error": UNKNOWN_MODEL_ERROR}


def test_rerank_for_an_unknown_model_is_404_like_ovms(reranker_app: TestClient) -> None:
    payload = {"model": EMBED_ALIAS, "query": "q", "documents": ["a"]}
    resp = reranker_app.post("/v3/rerank", json=payload)
    assert resp.status_code == 404
    assert resp.json() == {"error": UNKNOWN_MODEL_ERROR}


def test_role_specific_endpoints_and_readiness(
    embedder_app: TestClient, reranker_app: TestClient
) -> None:
    assert (
        embedder_app.get("/v3/models/snowflake-arctic-embed-l-v2.0").status_code == 200
    )
    assert embedder_app.get("/v3/models/other").status_code == 404
    assert (
        reranker_app.get("/v3/models/crossencoder-camembert-base-mmarcoFR").status_code
        == 200
    )
    assert embedder_app.post("/v3/rerank", json={}).status_code == 404
    assert reranker_app.post("/v3/embeddings", json={}).status_code == 404
    assert reranker_app.get("/healthz").json() == {"status": "ok", "role": "reranker"}


async def test_api_adapters_work_against_the_fastapi_server(
    settings_factory: Callable[..., RagSettings],
) -> None:
    """The contract test: the workflow's ``api`` adapters cannot tell the servers apart."""
    reranker_app = create_model_server(
        "reranker", RagPorts(settings_factory(), reranker=FakeReranker())
    )
    embedder_app = create_model_server(
        "embedder", RagPorts(settings_factory(), embedder=FakeEmbedder(dim=4))
    )
    with TestClient(reranker_app), TestClient(embedder_app):  # run the lifespans
        rerank_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=reranker_app), base_url="http://models"
        )
        embed_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=embedder_app), base_url="http://models"
        )
        reranker = api_reranker(rerank_client, "crossencoder-camembert-base-mmarcoFR")
        embedder = api_embedder(embed_client, "snowflake-arctic-embed-l-v2.0")
        passages = [
            ScoredPassage(id="x", text="cafeteria lunch", retrieval_score=0.9),
            ScoredPassage(id="y", text="vacation days policy", retrieval_score=0.8),
        ]
        ranked = await reranker.rerank("vacation days", passages, 1)
        assert [p.id for p in ranked] == ["y"]
        assert ranked[0].rerank_score is not None
        vectors = await embedder.embed(["a", "b"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        assert (await reranker.ready())[0]
        assert (await embedder.ready())[0]
        wrong_alias = api_reranker(rerank_client, "nope")
        assert not (await wrong_alias.ready())[0]


async def test_api_adapters_with_a_wrong_alias_fail_fast_on_404(
    settings_factory: Callable[..., RagSettings],
) -> None:
    """The unhappy path of the contract: unknown model -> 404, and 4xx is not retried."""
    sent: list[str] = []

    async def record_request(request: httpx.Request) -> None:
        sent.append(request.url.path)

    reranker_app = create_model_server(
        "reranker", RagPorts(settings_factory(), reranker=FakeReranker())
    )
    embedder_app = create_model_server(
        "embedder", RagPorts(settings_factory(), embedder=FakeEmbedder(dim=4))
    )
    with TestClient(reranker_app), TestClient(embedder_app):  # run the lifespans
        for app, call in [
            (reranker_app, _rerank_with_alias),
            (embedder_app, _embed_with_alias),
        ]:
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://models",
                event_hooks={"request": [record_request]},
            )
            sent.clear()
            with pytest.raises(RagDependencyError) as info:
                await call(client, "nope")
            assert info.value.detail == f'HTTP 404: {{"error":"{UNKNOWN_MODEL_ERROR}"}}'
            assert len(sent) == 1  # 4xx is not retried


async def _rerank_with_alias(client: httpx.AsyncClient, alias: str) -> None:
    reranker = api_reranker(client, alias)
    passages = [ScoredPassage(id="x", text="a", retrieval_score=0.9)]
    await reranker.rerank("q", passages, 1)


async def _embed_with_alias(client: httpx.AsyncClient, alias: str) -> None:
    embedder = api_embedder(client, alias)
    await embedder.embed(["a"])


def test_main_parses_role_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        "rag_load_test.entrypoints.rag_model_server.create_model_server",
        lambda role, ports: role,
    )
    monkeypatch.setattr(
        "rag_load_test.entrypoints.rag_model_server.uvicorn.run",
        lambda app, host, port: calls.update(app=app, host=host, port=port),
    )
    assert main(["--serve", "embedder", "--port", "8012"]) == 0
    assert calls == {"app": "embedder", "host": "0.0.0.0", "port": 8012}


def test_model_server_answers_under_the_domino_prefix(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., RagSettings]
) -> None:
    monkeypatch.setenv("DOMINO_RUN_HOST_PATH", "/apps/abc123")
    ports = RagPorts(settings_factory(), reranker=FakeReranker())
    with TestClient(create_model_server("reranker", ports)) as client:
        assert (
            client.get(
                "/apps/abc123/v3/models/crossencoder-camembert-base-mmarcoFR"
            ).status_code
            == 200
        )
        assert (
            client.get("/v3/models/crossencoder-camembert-base-mmarcoFR").status_code
            == 200
        )
