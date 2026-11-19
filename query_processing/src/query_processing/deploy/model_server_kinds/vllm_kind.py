"""The ``vllm`` model-server kind: vLLM's OpenAI-compatible server.

It serves an embedder, a reranker or an LLM from the model's published
weights, on vLLM's own /v1 routes: OpenAI's embeddings format, Cohere's rerank
format and OpenAI's chat completions, the formats the ``served`` backends
already speak. Its launch script is ``domino/model_servers/vllm.sh``, which
installs vLLM when the App's environment lacks it and runs ``vllm serve``: it
needs no bundle and no export, so the deploy tool exports nothing for it and
refuses no setting of it.

Example:
    >>> VllmModelServerKind().route_of("reranker")
    '/v1/rerank'
"""

from __future__ import annotations

from collections.abc import Mapping

from ...domain.endpoints.api_endpoint import (
    VLLM_CHAT_PATH,
    VLLM_EMBEDDINGS_PATH,
    VLLM_MODELS_ROUTE,
    VLLM_RERANK_PATH,
)
from .base_model_server_kind import MODEL_SERVER_KINDS, BaseModelServerKind


@MODEL_SERVER_KINDS.register("vllm")
class VllmModelServerKind(BaseModelServerKind):
    """vLLM: an embedder, a reranker or an LLM, each on its /v1 route.

    Example:
        >>> VllmModelServerKind().route_of("llm")
        '/v1/chat/completions'
    """

    @property
    def routes(self) -> Mapping[str, str]:
        """The /v1 route of each type vLLM serves."""
        return {
            "embedder": VLLM_EMBEDDINGS_PATH,
            "reranker": VLLM_RERANK_PATH,
            "llm": VLLM_CHAT_PATH,
        }

    @property
    def probe_route(self) -> str:
        """The list of vLLM's served models: it has no route for one model by name."""
        return VLLM_MODELS_ROUTE
