"""The ``fastapi`` model-server kind: ``qp-model-server``, our FastAPI model server.

It serves an embedder or a reranker from the component's ``local`` backend, on
OpenVINO Model Server's /v3 routes and formats, which it copies so that a client
cannot tell the two apart (``entrypoints/qp_model_server.py``). Its launch
script is ``domino/model_servers/fastapi.sh``.

Example:
    >>> FastApiModelServerKind().route_of("reranker")
    '/v3/rerank'
"""

from __future__ import annotations

from collections.abc import Mapping

from ...domain.api_endpoint import (
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_MODEL_ROUTE,
    MODEL_SERVER_RERANK_PATH,
)
from .base_model_server_kind import MODEL_SERVER_KINDS, BaseModelServerKind


@MODEL_SERVER_KINDS.register("fastapi")
class FastApiModelServerKind(BaseModelServerKind):
    """``qp-model-server``: an embedder or a reranker, on the /v3 routes.

    Example:
        >>> FastApiModelServerKind().probe_route
        '/v3/models/{name}'
    """

    @property
    def routes(self) -> Mapping[str, str]:
        """The /v3 route of each type the model server answers on."""
        return {
            "embedder": MODEL_SERVER_EMBEDDINGS_PATH,
            "reranker": MODEL_SERVER_RERANK_PATH,
        }

    @property
    def probe_route(self) -> str:
        """OVMS's model status route, which the model server answers too."""
        return MODEL_SERVER_MODEL_ROUTE
