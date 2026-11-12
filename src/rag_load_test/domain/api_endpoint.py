"""A model endpoint reached over HTTP, and the API formats it may speak.

An API format is a wire shape, not a vendor: any server answering OpenAI's
embeddings shape is spoken to as ``"openai"``, whoever runs it (OVMS, vLLM, a
gateway, OpenAI itself). Adding a format means adding its name to the matching
``Literal`` here and its module and registry entry in ``adapters/model_api``.

Example:
    >>> endpoint = ApiEndpoint(url="http://host:8000/v3/embeddings",
    ...                        api_format="openai", api_key="", model="bge-small")
"""

from __future__ import annotations

from typing import Literal, NamedTuple

EmbeddingApiFormat = Literal["openai"]
RerankApiFormat = Literal["cohere"]
GenerationApiFormat = Literal["openai"]

# The ``api_key`` that means "send the Domino run's token", fetched per run.
DOMINO_RUN_TOKEN = "domino"

# The routes our model servers answer on: OpenVINO Model Server's, which the
# FastAPI model server copies so a client cannot tell the two apart.
MODEL_SERVER_EMBEDDINGS_PATH = "/v3/embeddings"
MODEL_SERVER_RERANK_PATH = "/v3/rerank"
MODEL_SERVER_MODEL_ROUTE = "/v3/models/{name}"


class ApiEndpoint(NamedTuple):
    """Where one model is served, in which API format, and how to authenticate.

    Attributes:
        url: The full route, such as ``http://host:8000/v3/embeddings``.
        api_format: The wire shape the route speaks, such as ``"openai"``.
        api_key: "" sends no ``Authorization`` header; ``DOMINO_RUN_TOKEN`` sends
            the Domino run's token; anything else is sent as a bearer token.
        model: The model name sent with every request.

    Example:
        >>> ApiEndpoint("https://api.cohere.com/v2/rerank", "cohere", "key",
        ...             "rerank-v3.5")
    """

    url: str
    api_format: str
    api_key: str
    model: str
