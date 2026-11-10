"""The FastAPI model server must be indistinguishable from OVMS for the ``ovms`` adapters."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_load_test.fakes import FakeEmbedder, FakeReranker
from rag_load_test.model_server import create_model_server, main
from rag_load_test.models import ScoredPassage
from rag_load_test.ovms import BearerToken, OvmsEmbedder, OvmsReranker
from rag_load_test.settings import RagSettings

LOCUSTFILE = Path(__file__).resolve().parents[1] / "locustfile.py"


@pytest.fixture
def embedder_app(settings_factory: Callable[..., RagSettings]):
    app = create_model_server(
        "embedder", settings_factory(), embedder=FakeEmbedder(dim=4)
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def reranker_app(settings_factory: Callable[..., RagSettings]):
    app = create_model_server("reranker", settings_factory(), reranker=FakeReranker())
    with TestClient(app) as client:
        yield client


def test_embeddings_endpoint_matches_ovms_shape(embedder_app: TestClient) -> None:
    resp = embedder_app.post(
        "/v3/embeddings", json={"model": "bge-small-en-v1.5", "input": ["a b", "c d"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list" and body["model"] == "bge-small-en-v1.5"
    assert [row["index"] for row in body["data"]] == [0, 1]
    assert all(len(row["embedding"]) == 4 for row in body["data"])


def test_embeddings_accepts_a_single_string(embedder_app: TestClient) -> None:
    body = embedder_app.post(
        "/v3/embeddings", json={"model": "m", "input": "hi"}
    ).json()
    assert len(body["data"]) == 1


def test_rerank_endpoint_matches_ovms_shape(reranker_app: TestClient) -> None:
    payload = {
        "model": "bge-reranker-base",
        "query": "vacation days",
        "documents": ["cafeteria lunch", "vacation days policy", "unrelated"],
        "top_n": 2,
    }
    body = reranker_app.post("/v3/rerank", json=payload).json()
    assert [r["index"] for r in body["results"]] == [1, 0]
    scores = [r["relevance_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_without_top_n_returns_every_document(reranker_app: TestClient) -> None:
    payload = {"model": "m", "query": "q", "documents": ["a", "b", "c"]}
    assert len(reranker_app.post("/v3/rerank", json=payload).json()["results"]) == 3


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
        "reranker", settings_factory(), reranker=FakeReranker()
    )
    embedder_app = create_model_server(
        "embedder", settings_factory(), embedder=FakeEmbedder(dim=4)
    )
    with TestClient(reranker_app), TestClient(embedder_app):  # run the lifespans
        rerank_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=reranker_app), base_url="http://models"
        )
        embed_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=embedder_app), base_url="http://models"
        )
        reranker = OvmsReranker(
            rerank_client,
            "http://models",
            "bge-reranker-base",
            token=BearerToken(rerank_client, ""),
        )
        embedder = OvmsEmbedder(
            embed_client,
            "http://models",
            "bge-small-en-v1.5",
            token=BearerToken(embed_client, ""),
        )
        passages = [
            ScoredPassage(id="x", text="cafeteria lunch", retrieval_score=0.9),
            ScoredPassage(id="y", text="vacation days policy", retrieval_score=0.8),
        ]
        ranked = await reranker.rerank("vacation days", passages, 1)
        assert [p.id for p in ranked] == ["y"] and ranked[0].rerank_score is not None
        vectors = await embedder.embed(["a", "b"])
        assert len(vectors) == 2 and len(vectors[0]) == 4
        assert (await reranker.ready())[0] and (await embedder.ready())[0]
        assert not (
            await OvmsReranker(
                rerank_client,
                "http://models",
                "nope",
                token=BearerToken(rerank_client, ""),
            ).ready()
        )[0]


def test_main_parses_role_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        "rag_load_test.model_server.create_model_server", lambda role, s: role
    )
    monkeypatch.setattr(
        "rag_load_test.model_server.uvicorn.run",
        lambda app, host, port: calls.update(app=app, host=host, port=port),
    )
    assert main(["--serve", "embedder", "--port", "8012"]) == 0
    assert calls == {"app": "embedder", "host": "0.0.0.0", "port": 8012}


def _load_locustfile(name: str):
    spec = importlib.util.spec_from_file_location(name, LOCUSTFILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("target", "active"),
    [("rag", "RagUser"), ("rerank", "RerankUser"), ("embeddings", "EmbeddingsUser")],
)
def test_loadtest_target_selects_one_user_class(
    monkeypatch: pytest.MonkeyPatch, target: str, active: str
) -> None:
    monkeypatch.setenv("RAG_LOADTEST_TARGET", target)
    module = _load_locustfile(f"locustfile_target_{target}")
    flags = {
        name: getattr(module, name).abstract
        for name in ("RagUser", "RerankUser", "EmbeddingsUser")
    }
    assert [name for name, abstract in flags.items() if not abstract] == [active]
    assert len(module.RERANK_DOCUMENTS) >= 20


def test_model_server_answers_under_the_domino_prefix(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., RagSettings]
) -> None:
    monkeypatch.setenv("DOMINO_RUN_HOST_PATH", "/apps/abc123")
    app = create_model_server("reranker", settings_factory(), reranker=FakeReranker())
    with TestClient(app) as client:
        assert client.get("/apps/abc123/v3/models/bge-reranker-base").status_code == 200
        assert client.get("/v3/models/bge-reranker-base").status_code == 200
