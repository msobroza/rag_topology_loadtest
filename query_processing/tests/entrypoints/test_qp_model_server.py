"""The FastAPI model server must be indistinguishable from OVMS for the ``api`` adapters.

The wire first, through a test client; then the functions that build its answers,
pure: no HTTP, no server.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient
from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from model_api_fakes import endpoint_client
from query_processing.adapters.embedder.api.api_embedder import ApiEmbedder
from query_processing.adapters.embedder.api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)
from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.reranker.api.api_reranker import ApiReranker
from query_processing.adapters.reranker.api.cohere_rerank_format import (
    CohereRerankFormat,
)
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.endpoints.api_endpoint import ApiEndpoint
from query_processing.domain.endpoints.standard_formats import CohereRerankRequest
from query_processing.domain.errors import RagDependencyError
from query_processing.entrypoints.qp_model_server import (
    UNKNOWN_MODEL_ERROR,
    candidates_from_rerank,
    create_model_server,
    embeddings_response_for,
    main,
    rerank_response_for,
)
from sentence_transformers_fakes import FakeSentenceTransformersModule

EMBED_ALIAS = "fixture-embedder"  # the embedder's alias in the workflow file
RERANK_ALIAS = "fixture-reranker"  # the reranker's


def api_reranker(client: httpx.AsyncClient, alias: str) -> ApiReranker:
    """An ``ApiReranker`` at ``http://models/v3/rerank`` without auth, no sleeping."""
    endpoint = ApiEndpoint("http://models/v3/rerank", "cohere", "", alias)
    rerank_client = endpoint_client(client, endpoint, component="reranker")
    return ApiReranker(rerank_client, CohereRerankFormat())


def api_embedder(client: httpx.AsyncClient, alias: str) -> ApiEmbedder:
    """An ``ApiEmbedder`` at ``http://models/v3/embeddings`` without auth."""
    endpoint = ApiEndpoint("http://models/v3/embeddings", "openai", "", alias)
    embed_client = endpoint_client(client, endpoint, component="embedder")
    return ApiEmbedder(
        embed_client, OpenAIEmbeddingsFormat(), batch_size=64, query_prefix=""
    )


@pytest.fixture
def embedder_app(settings_factory: Callable[..., WorkflowSettings]):
    composition = CompositionRoot(settings_factory(), embedder=FakeEmbedder(dim=4))
    with TestClient(create_model_server("embedder", composition)) as client:
        yield client


@pytest.fixture
def reranker_app(settings_factory: Callable[..., WorkflowSettings]):
    composition = CompositionRoot(settings_factory(), reranker=FakeReranker())
    with TestClient(create_model_server("reranker", composition)) as client:
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
    assert embedder_app.get("/v3/models/fixture-embedder").status_code == 200
    assert embedder_app.get("/v3/models/other").status_code == 404
    assert reranker_app.get("/v3/models/fixture-reranker").status_code == 200
    assert embedder_app.post("/v3/rerank", json={}).status_code == 404
    assert reranker_app.post("/v3/embeddings", json={}).status_code == 404
    assert reranker_app.get("/healthz").json() == {"status": "ok", "role": "reranker"}


async def test_api_adapters_work_against_the_fastapi_server(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    """The contract test: the workflow's ``api`` adapters cannot tell the servers apart."""
    reranker_app = create_model_server(
        "reranker", CompositionRoot(settings_factory(), reranker=FakeReranker())
    )
    embedder_app = create_model_server(
        "embedder", CompositionRoot(settings_factory(), embedder=FakeEmbedder(dim=4))
    )
    with TestClient(reranker_app), TestClient(embedder_app):  # run the lifespans
        rerank_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=reranker_app), base_url="http://models"
        )
        embed_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=embedder_app), base_url="http://models"
        )
        reranker = api_reranker(rerank_client, "fixture-reranker")
        embedder = api_embedder(embed_client, "fixture-embedder")
        records = [
            ScoredVectorStoreRecord(
                text="cafeteria lunch", metadata={"id": "x"}, retrieval_score=0.9
            ),
            ScoredVectorStoreRecord(
                text="vacation days policy", metadata={"id": "y"}, retrieval_score=0.8
            ),
        ]
        ranked = await reranker.rerank("vacation days", records, 1)
        assert [p.id for p in ranked] == ["y"]
        assert ranked[0].rerank_score is not None
        vectors = await embedder.embed_documents(["a", "b"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        assert (await reranker.ready())[0]
        assert (await embedder.ready())[0]
        wrong_alias = api_reranker(rerank_client, "nope")
        assert not (await wrong_alias.ready())[0]


async def test_api_adapters_with_a_wrong_alias_fail_fast_on_404(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    """The unhappy path of the contract: unknown model -> 404, and 4xx is not retried."""
    sent: list[str] = []

    async def record_request(request: httpx.Request) -> None:
        sent.append(request.url.path)

    reranker_app = create_model_server(
        "reranker", CompositionRoot(settings_factory(), reranker=FakeReranker())
    )
    embedder_app = create_model_server(
        "embedder", CompositionRoot(settings_factory(), embedder=FakeEmbedder(dim=4))
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
    records = [
        ScoredVectorStoreRecord(text="a", metadata={"id": "x"}, retrieval_score=0.9)
    ]
    await reranker.rerank("q", records, 1)


async def _embed_with_alias(client: httpx.AsyncClient, alias: str) -> None:
    embedder = api_embedder(client, alias)
    await embedder.embed_documents(["a"])


def test_main_parses_role_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        "query_processing.entrypoints.qp_model_server.create_model_server",
        lambda role, composition: role,
    )
    monkeypatch.setattr(
        "query_processing.entrypoints.qp_model_server.uvicorn.run",
        lambda app, host, port, **_: calls.update(app=app, host=host, port=port),
    )
    assert main(["--serve", "embedder", "--port", "8012"]) == 0
    assert calls == {"app": "embedder", "host": "0.0.0.0", "port": 8012}


def test_a_component_of_any_name_is_served_under_its_alias(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # Its weights are a folder, its alias the last part; served whatever its backend.
    coarse = {"type": "reranker", "backend": "fake", "source_id": "/models/coarse-rr/"}
    settings = settings_factory(components={"coarse": coarse})
    composition = CompositionRoot(settings, coarse=FakeReranker())

    with TestClient(create_model_server("coarse", composition)) as client:
        status = client.get("/v3/models/coarse-rr")
        other = client.get(f"/v3/models/{RERANK_ALIAS}")
        healthz = client.get("/healthz").json()

    assert status.json() == {"name": "coarse-rr", "role": "coarse"}
    assert other.status_code == 404
    assert healthz == {"status": "ok", "role": "coarse"}


def test_the_named_component_is_served_from_its_local_backend_whatever_its_file_says(
    settings_factory: Callable[..., WorkflowSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Its weights load in this process (the stand-in records it), although the
    # workflow file serves it from a model server: that model server is this one.
    weights = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", weights)
    coarse = {
        "type": "reranker",
        "backend": "served",
        "source_id": "org/coarse-rr",
        "served": {"url": "http://models.test/v3/rerank"},
        "local": {"threads": 1},  # pinned: no calibration
    }
    settings = settings_factory(
        components={"coarse": coarse}, local_models__runtime="torch"
    )
    payload = {"model": "coarse-rr", "query": "q", "documents": ["a", "b"]}

    with TestClient(create_model_server("coarse", CompositionRoot(settings))) as client:
        answer = client.post("/v3/rerank", json=payload)

    assert answer.status_code == 200, answer.text
    assert len(answer.json()["results"]) == 2
    assert weights.loaded == ["reranker:org/coarse-rr"]


def test_a_component_no_model_server_serves_is_refused(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    composition = CompositionRoot(settings_factory())

    with pytest.raises(ValueError, match="no component 'llm' a model server can"):
        create_model_server("llm", composition)


@pytest.mark.parametrize("name", ["llm", "gpu"])
def test_main_refuses_a_name_it_cannot_serve_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    monkeypatch.setattr(
        "query_processing.entrypoints.qp_model_server.uvicorn.run",
        lambda app, **_: pytest.fail("nothing is served"),
    )

    with pytest.raises(SystemExit) as exited:
        main(["--serve", name])

    assert exited.value.code == 2
    assert "expected an embedder or a reranker of the workflow file: embedder, " in (
        capsys.readouterr().err
    )


def test_model_server_answers_under_the_domino_prefix(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., WorkflowSettings]
) -> None:
    monkeypatch.setenv("DOMINO_RUN_HOST_PATH", "/apps/abc123")
    composition = CompositionRoot(settings_factory(), reranker=FakeReranker())
    with TestClient(create_model_server("reranker", composition)) as client:
        assert client.get("/apps/abc123/v3/models/fixture-reranker").status_code == 200
        assert client.get("/v3/models/fixture-reranker").status_code == 200


# --- the answers, pure: no HTTP, no server ---------------------------------


def test_unknown_model_error_is_the_ovms_text() -> None:
    expected = "Mediapipe graph definition with requested name is not found"
    assert expected == UNKNOWN_MODEL_ERROR


def test_embeddings_response_for_numbers_rows_in_input_order() -> None:
    response = embeddings_response_for("e", [[0.1, 0.2], [0.3, 0.4]])
    assert response.model_dump() == {
        "object": "list",
        "model": "e",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
    }
    body = response.model_dump()
    assert OpenAIEmbeddingsFormat().vectors(body) == [[0.1, 0.2], [0.3, 0.4]]


def test_candidates_from_rerank_carry_their_document_index_as_id() -> None:
    request = CohereRerankRequest(model="r", query="q", documents=["a", "b"])
    candidates = candidates_from_rerank(request)
    assert [(p.id, p.text) for p in candidates] == [("0", "a"), ("1", "b")]
    assert all(p.rerank_score is None for p in candidates)
    assert all(p.retrieval_score == 0.0 for p in candidates)


def test_rerank_response_for_names_documents_by_index() -> None:
    request = CohereRerankRequest(model="r", query="q", documents=["a", "b", "c"])
    candidates = candidates_from_rerank(request)
    ranked = [
        candidates[2].model_copy(update={"rerank_score": 0.9}),
        candidates[0].model_copy(update={"rerank_score": 0.4}),
    ]
    assert rerank_response_for(ranked).model_dump() == {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ]
    }


def test_rerank_round_trip_maps_back_to_the_documents_sent() -> None:
    api_format = CohereRerankFormat()
    body = api_format.request_body("r", "q", ["a", "b", "c"], top_n=2)
    candidates = candidates_from_rerank(CohereRerankRequest.model_validate(body))
    ranked = [candidates[1].model_copy(update={"rerank_score": 0.8})]
    answer = rerank_response_for(ranked).model_dump()
    assert api_format.scores(answer) == {1: 0.8}


def test_rerank_response_for_rejects_an_unscored_record() -> None:
    [unscored] = candidates_from_rerank(
        CohereRerankRequest(model="r", query="q", documents=["a"])
    )
    with pytest.raises(ValueError, match="'0' has no rerank_score"):
        rerank_response_for([unscored])
