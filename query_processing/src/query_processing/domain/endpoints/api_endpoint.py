"""A model endpoint reached over HTTP, and the routes our model servers answer on.

An API format is a wire shape, not a vendor: any server answering OpenAI's
embeddings shape is spoken to as ``"openai"``, whoever runs it (OVMS, vLLM, a
gateway, OpenAI itself). The formats are registered by name, one module each,
in the ``api`` package of each component under ``adapters/``; an endpoint
names one, and a command refuses an unknown name before it opens anything.

Example:
    >>> endpoint = ApiEndpoint(url="http://host:8000/v3/embeddings",
    ...                        api_format="openai", model="bge-small")
"""

from __future__ import annotations

from typing import Final, NamedTuple

# The value of an endpoint's ``api_key`` setting that means "send the Domino
# run's token": the composition root gives that endpoint the ``domino`` key source.
DOMINO_RUN_TOKEN = "domino"
# The backend that calls a model server of ours, which every component type
# has: named here for the composition and the deploy tool, which share no other
# package.
SERVED_BACKEND: Final = "served"

# The routes our model servers answer on: OpenVINO Model Server's, which the
# FastAPI model server copies so a client cannot tell the two apart.
MODEL_SERVER_EMBEDDINGS_PATH = "/v3/embeddings"
MODEL_SERVER_RERANK_PATH = "/v3/rerank"
# OVMS's chat completions route, on which the ``ovms`` kind serves an LLM.
MODEL_SERVER_CHAT_PATH = "/v3/chat/completions"
MODEL_SERVER_MODEL_ROUTE = "/v3/models/{name}"
# vLLM's routes (its OpenAI-compatible server, 0.32.0), on which the ``vllm``
# kind serves each model: OpenAI's embeddings format, Cohere's rerank format,
# OpenAI's chat completions. It lists its served models at /v1/models.
VLLM_EMBEDDINGS_PATH = "/v1/embeddings"
VLLM_RERANK_PATH = "/v1/rerank"
VLLM_CHAT_PATH = "/v1/chat/completions"
VLLM_MODELS_ROUTE = "/v1/models"
# The call route: a component type with no outside API format, its request in
# and its answer out in its API format (``domain/endpoints/api_format.py``); the FastAPI
# model server's alone.
MODEL_SERVER_CALL_PATH = "/call/{component_type}"


def call_path(component_type: str) -> str:
    """Return the call route a model server serves a component type on.

    Args:
        component_type: The component type's name, as it registers.

    Returns:
        The path, which a ``served`` block's URL ends with.

    Example:
        >>> call_path("shouter")
        '/call/shouter'
    """
    return MODEL_SERVER_CALL_PATH.format(component_type=component_type)


class ApiEndpoint(NamedTuple):
    """Where one model is served, and in which API format; its key is its key source's.

    Attributes:
        url: The full route, such as ``http://host:8000/v3/embeddings``.
        api_format: The wire shape the route speaks, such as ``"openai"``.
        model: The model name sent with every request.

    Example:
        >>> ApiEndpoint("https://api.cohere.com/v2/rerank", "cohere", "rerank-v3.5")
    """

    url: str
    api_format: str
    model: str
