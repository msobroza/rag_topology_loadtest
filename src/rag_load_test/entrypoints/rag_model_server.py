"""A FastAPI model server on OVMS's routes and formats (the FastAPI-vs-OVMS experiment).

Like an OVMS app it serves ONE model per process: ``--serve NAME`` names a
component of the workflow file, always opened on its ``local`` backend. An
embedder exposes ``POST /v3/embeddings`` (OpenAI's embeddings format), a
reranker ``POST /v3/rerank`` (Cohere's rerank format). Both answer
``GET /v3/models/{name}``, as OVMS does. Routes come from ``domain/api_endpoint.py``
and bodies are the standard formats' own models (``OpenAIEmbeddingsRequest``,
``CohereRerankRequest`` of ``domain/standard_formats.py``), the shapes the ``api``
adapters speak, so client and server cannot drift apart; the answers are built
by the three functions at the end of this module. It serves its model under
the component's alias, and loads the weights its source id names; like OVMS, a
request for any model but this server's alias is a 404. The workflow app cannot
tell the two servers apart, so ``RAG_COMPONENTS__RERANKER__API__URL`` can point
at either and the same load test compares FastAPI with OVMS.

Example:
    $ rag-model-server --serve reranker --port 8011
    $ RAG_TOPOLOGY=split-reranker \
    >   RAG_COMPONENTS__RERANKER__API__URL=http://localhost:8011/v3/rerank rag-serve
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from http import HTTPStatus

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from ..composition.composition_root import CompositionRoot
from ..composition.registered_names import check_registered_names
from ..config.rag_settings import RagSettings
from ..domain.api_endpoint import (
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_MODEL_ROUTE,
    MODEL_SERVER_RERANK_PATH,
)
from ..domain.passages import ScoredPassage
from ..domain.standard_formats import (
    CohereRerankRequest,
    CohereRerankResponse,
    CohereRerankResult,
    OpenAIEmbeddingRow,
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
)

# Verbatim from OVMS: the body of its 404 when ``model`` names no served model.
UNKNOWN_MODEL_ERROR = "Mediapipe graph definition with requested name is not found"


# The handlers read what the factory left on ``app.state``: ``part`` (the
# component's name), ``alias`` and ``composition`` from the start, the served
# model once the lifespan has loaded it.
common_routes = APIRouter()
embedder_routes = APIRouter()
reranker_routes = APIRouter()
# The inference route of each component type a model server serves.
_ROUTES_BY_TYPE = {"embedder": embedder_routes, "reranker": reranker_routes}


def servable_components(settings: RagSettings) -> list[str]:
    """Return the components a model server can serve: the embedders and rerankers.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> servable_components(RagSettings())
        ['embedder', 'reranker']
    """
    return [n for n, c in settings.components.items() if c.type in _ROUTES_BY_TYPE]


def create_model_server(name: str, composition: CompositionRoot) -> FastAPI:
    """Build the server for one component; it always serves its local model.

    The served name is the component's alias, the name the clients send; the
    weights are those its source id names.

    Args:
        name: The component this process serves, an embedder or a reranker; its
            type picks the one inference route.
        composition: Builds the in-process model when the lifespan starts.

    Returns:
        The app, its routes answering under ``DOMINO_RUN_HOST_PATH`` and at the root.

    Raises:
        ValueError: If ``name`` is no embedder or reranker of the workflow file.

    Example:
        >>> composition = CompositionRoot(settings, reranker=FakeReranker())
        >>> create_model_server("reranker", composition)
    """
    servable = servable_components(composition.settings)
    if name not in servable:
        raise ValueError(
            f"no component {name!r} a model server can serve; expected an "
            f"embedder or a reranker of the workflow file: {', '.join(servable)}"
        )
    component = composition.settings.components[name]
    app = FastAPI(
        title=f"rag-model-server ({name})",
        root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""),  # Domino's prefix
        lifespan=_lifespan,
    )
    app.state.part, app.state.composition = name, composition
    app.state.alias = component.alias
    app.include_router(common_routes)
    app.include_router(_ROUTES_BY_TYPE[component.type])
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    composition: CompositionRoot = app.state.composition
    name = app.state.part
    async with composition:
        if composition.settings.components[name].type == "embedder":
            app.state.embedder = await composition.local_embedder(name)
        else:
            app.state.reranker = await composition.local_reranker(name)
        yield


@common_routes.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    """Answer while the process is up, naming the model it serves.

    Args:
        request: The HTTP request, for the app's part.

    Returns:
        ``{"status": "ok", "role": ...}``: the component's name, under the
        key the route has always used.
    """
    return {"status": "ok", "role": request.app.state.part}


@common_routes.get(MODEL_SERVER_MODEL_ROUTE)
async def model_status(name: str, request: Request) -> JSONResponse:
    """Answer OVMS's model status route for the named model.

    Args:
        name: The model the client asks about.
        request: The HTTP request, for the app's alias and part.

    Returns:
        200 naming the alias and part (under ``role``), or 404 when the name is
        not the alias.
    """
    alias = request.app.state.alias
    if name != alias:
        body = {"error": f"unknown model {name!r}, this server serves {alias!r}"}
        return JSONResponse(status_code=HTTPStatus.NOT_FOUND, content=body)
    content = {"name": alias, "role": request.app.state.part}
    return JSONResponse(status_code=HTTPStatus.OK, content=content)


@embedder_routes.post(MODEL_SERVER_EMBEDDINGS_PATH)
async def embeddings(body: OpenAIEmbeddingsRequest, request: Request) -> JSONResponse:
    """Embed the texts of the request, as OVMS does.

    Args:
        body: The model alias and the texts (one bare string reads as one text).
        request: The HTTP request, for the app's alias and embedder.

    Returns:
        One vector per text, or OVMS's 404 when the model is not the alias.
    """
    alias = request.app.state.alias
    if body.model != alias:
        return _unknown_model()
    # As documents: a client of ours has already put the query prefix in front of
    # a query, so adding it here would prefix that query twice.
    vectors = await request.app.state.embedder.embed_documents(body.input)
    return JSONResponse(embeddings_response_for(alias, vectors).model_dump())


@reranker_routes.post(MODEL_SERVER_RERANK_PATH)
async def rerank(body: CohereRerankRequest, request: Request) -> JSONResponse:
    """Rerank the documents of the request against its query, as OVMS does.

    Args:
        body: The model alias, the query, the documents and an optional top_n.
        request: The HTTP request, for the app's alias and reranker.

    Returns:
        The top_n documents (all when unset) by score, or OVMS's 404 when the
        model is not the alias.
    """
    if body.model != request.app.state.alias:
        return _unknown_model()
    passages = candidates_from_rerank(body)
    top_n = body.top_n or len(passages)
    ranked = await request.app.state.reranker.rerank(body.query, passages, top_n)
    return JSONResponse(rerank_response_for(ranked).model_dump())


def _unknown_model() -> JSONResponse:
    # What OVMS answers when ``model`` names no served model (checked in its source).
    return JSONResponse(
        status_code=HTTPStatus.NOT_FOUND, content={"error": UNKNOWN_MODEL_ERROR}
    )


# --- the answers, in the standard formats' shapes ---------------------------------


def embeddings_response_for(
    alias: str, vectors: Sequence[Sequence[float]]
) -> OpenAIEmbeddingsResponse:
    """Build the embeddings answer.

    Args:
        alias: The name the embedder is served under, sent back as ``model``.
        vectors: One vector per input text, in input order.

    Returns:
        The answer, where row ``index`` i holds the vector of input text i.

    Example:
        >>> embeddings_response_for("embedder-alias", [[0.1, 0.2]]).data[0].index
        0
    """
    rows = [
        OpenAIEmbeddingRow(index=i, embedding=list(vector))
        for i, vector in enumerate(vectors)
    ]
    return OpenAIEmbeddingsResponse(model=alias, data=rows)


def candidates_from_rerank(request: CohereRerankRequest) -> list[ScoredPassage]:
    """Turn the request's documents into rerank candidates.

    Args:
        request: The rerank request the server received.

    Returns:
        One passage per document, whose ``id`` is its index in the request and
        whose ``retrieval_score`` is 0.0.

    Example:
        >>> candidates_from_rerank(request)[1].id
        '1'
    """
    return [
        ScoredPassage(id=str(i), text=text, retrieval_score=0.0)
        for i, text in enumerate(request.documents)
    ]


def rerank_response_for(ranked: Sequence[ScoredPassage]) -> CohereRerankResponse:
    """Build the rerank answer.

    Args:
        ranked: Reranked passages made by ``candidates_from_rerank``, best first;
            each must carry a ``rerank_score``.

    Returns:
        One result per passage, in the given order, ``index`` taken from its id.

    Raises:
        ValueError: If a passage has no ``rerank_score``.

    Example:
        >>> rerank_response_for(await reranker.rerank(query, candidates, top_n))
    """
    return CohereRerankResponse(results=[_rerank_result(p) for p in ranked])


def _rerank_result(passage: ScoredPassage) -> CohereRerankResult:
    if passage.rerank_score is None:
        raise ValueError(
            f"passage {passage.id!r} has no rerank_score; expected a reranked "
            "passage from candidates_from_rerank"
        )
    return CohereRerankResult(
        index=int(passage.id), relevance_score=passage.rerank_score
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-model-server --serve NAME [--host H] [--port P]``.

    Serves the one component with uvicorn, on 0.0.0.0:8888 unless told
    otherwise, with a composition root built from the ``RAG_*`` settings. A
    settings mistake, an unknown backend or API format included, and a name
    that is no embedder or reranker exit as a usage error.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status, 0, once the server stops.
    """
    parser = argparse.ArgumentParser(
        prog="rag-model-server", description="Serve one model."
    )
    parser.add_argument(
        "--serve",
        required=True,
        metavar="NAME",
        help="the embedder or reranker component to serve, such as reranker",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args(argv)
    try:
        settings = RagSettings()
        check_registered_names(settings)
        app = create_model_server(args.serve, CompositionRoot(settings))
    except ValueError as exc:  # a settings mistake is a usage error, not a crash
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
