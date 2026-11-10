"""The FastAPI model server must be indistinguishable from OVMS for the ``ovms`` adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_load_test.adapters.fakes import FakeEmbedder, FakeReranker
from rag_load_test.adapters.http_policy import HttpPolicy
from rag_load_test.adapters.ovms import BearerToken, OvmsEmbedder, OvmsReranker
from rag_load_test.adapters.rag_ports import RagPorts
from rag_load_test.adapters.v3_contract import UNKNOWN_MODEL_ERROR
from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import ScoredPassage
from rag_load_test.entrypoints.model_server import create_model_server, main

EMBED_ALIAS = "bge-small-en-v1.5"  # RagSettings().ovms_embeddings_model
RERANK_ALIAS = "bge-reranker-base"  # RagSettings().ovms_rerank_model


def ovms_reranker(client: httpx.AsyncClient, alias: str) -> OvmsReranker:
    """An ``OvmsReranker`` at ``http://models`` without auth, retries not sleeping."""
    return OvmsReranker(client, "http://models", alias, **_no_auth(client))


def ovms_embedder(client: httpx.AsyncClient, alias: str) -> OvmsEmbedder:
    """An ``OvmsEmbedder`` at ``http://models`` without auth, retries not sleeping."""
    return OvmsEmbedder(
        client, "http://models", alias, batch_size=64, **_no_auth(client)
    )


def _no_auth(client: httpx.AsyncClient) -> dict[str, Any]:
    policy = HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.0)
    token = BearerToken(
        client, "", token_url="http://unused.test", ttl_s=240.0, policy=policy
    )
    return {"token": token, "policy": policy}


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
    assert body["object"] == "list" and body["model"] == EMBED_ALIAS
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
    assert embedder_app.get("/v3/models/bge-small-en-v1.5").status_code == 200
    assert embedder_app.get("/v3/models/other").status_code == 404
    assert reranker_app.get("/v3/models/bge-reranker-base").status_code == 200
    assert embedder_app.post("/v3/rerank", json={}).status_code == 404
    assert reranker_app.post("/v3/embeddings", json={}).status_code == 404
    assert reranker_app.get("/healthz").json() == {"status": "ok", "role": "reranker"}


async def test_ovms_adapters_work_against_the_fastapi_server(
    settings_factory: Callable[..., RagSettings],
) -> None:
    """The contract test: the workflow's OVMS adapters cannot tell the servers apart."""
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
        reranker = ovms_reranker(rerank_client, "bge-reranker-base")
        embedder = ovms_embedder(embed_client, "bge-small-en-v1.5")
        passages = [
            ScoredPassage(id="x", text="cafeteria lunch", retrieval_score=0.9),
            ScoredPassage(id="y", text="vacation days policy", retrieval_score=0.8),
        ]
        ranked = await reranker.rerank("vacation days", passages, 1)
        assert [p.id for p in ranked] == ["y"] and ranked[0].rerank_score is not None
        vectors = await embedder.embed(["a", "b"])
        assert len(vectors) == 2 and len(vectors[0]) == 4
        assert (await reranker.ready())[0] and (await embedder.ready())[0]
        wrong_alias = ovms_reranker(rerank_client, "nope")
        assert not (await wrong_alias.ready())[0]


async def test_ovms_adapters_with_a_wrong_alias_fail_fast_on_404(
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
    reranker = ovms_reranker(client, alias)
    passages = [ScoredPassage(id="x", text="a", retrieval_score=0.9)]
    await reranker.rerank("q", passages, 1)


async def _embed_with_alias(client: httpx.AsyncClient, alias: str) -> None:
    embedder = ovms_embedder(client, alias)
    await embedder.embed(["a"])


def test_main_parses_role_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        "rag_load_test.entrypoints.model_server.create_model_server",
        lambda role, ports: role,
    )
    monkeypatch.setattr(
        "rag_load_test.entrypoints.model_server.uvicorn.run",
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
        assert client.get("/apps/abc123/v3/models/bge-reranker-base").status_code == 200
        assert client.get("/v3/models/bge-reranker-base").status_code == 200
