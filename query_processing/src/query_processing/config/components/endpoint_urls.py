"""The URL of a component called over HTTP, and the refusal of a missing one.

A model component on ``served`` or ``api`` is called at the URL of the block
named after its backend, and a component of a registered type (the router, or
a type an extension defines) on ``served`` at the URL of its ``served`` block,
its model server's call route. ``WorkflowSettings`` refuses one that does not
start with ``http://`` or ``https://`` at start-up, naming the variables to set
and an example of the route.

Example:
    >>> component_endpoint_url(settings.components["llm"])
    'https://api.openai.com/v1/chat/completions'
"""

from __future__ import annotations

from ...domain.endpoints.api_endpoint import SERVED_BACKEND, call_path
from .component_settings import EmbedderSettings, LlmSettings, RerankerSettings
from .registered_component_settings import RegisteredComponentSettings
from .settings_blocks import ComponentSettings

# Shown in the error for a backend called over HTTP without a URL: one route
# per backend and type, a model server of ours or an endpoint we do not run.
_URL_EXAMPLES = {
    SERVED_BACKEND: {
        "embedder": "http://localhost:8002/v3/embeddings",
        "reranker": "http://localhost:8001/v3/rerank",
        "llm": "http://localhost:8000/v3/chat/completions",
    },
    "api": {
        "embedder": "https://api.openai.com/v1/embeddings",
        "reranker": "https://api.cohere.com/v2/rerank",
        "llm": "https://api.openai.com/v1/chat/completions",
    },
}
# Where the FastAPI model server listens in the example of a call route.
_CALL_ROUTE_HOST = "http://localhost:8888"
_MODELS = (EmbedderSettings, RerankerSettings, LlmSettings)


def component_endpoint_url(component: ComponentSettings) -> str | None:
    """Return the URL a component is called at, from the block of its backend.

    Args:
        component: Any component of the workflow file.

    Returns:
        The URL of a model's ``served`` or ``api`` block, or of a registered
        type's ``served`` block; None for a backend that calls no endpoint.

    Example:
        >>> component_endpoint_url(EmbedderSettings(backend="fake", source_id="o/m"))
        None
    """
    if component.backend == SERVED_BACKEND and isinstance(
        component, (*_MODELS, RegisteredComponentSettings)
    ):
        return component.served.url
    if component.backend == "api" and isinstance(component, _MODELS):
        return component.api.url
    return None


def missing_url_message(
    component: ComponentSettings, backend_variable: str, url_variable: str, url: str
) -> str:
    """Return the refusal of a component whose URL is not one.

    Args:
        component: The component, on ``served`` or ``api``.
        backend_variable: The variable of its backend.
        url_variable: The variable of its URL.
        url: The URL found.

    Returns:
        The variables, the value and an example: the same shape for every component.

    Example:
        >>> missing_url_message(llm, "QP_COMPONENTS__LLM__BACKEND",
        ...                     "QP_COMPONENTS__LLM__API__URL", "")
        "QP_COMPONENTS__LLM__BACKEND='api' needs QP_COMPONENTS__LLM__API__URL, ..."
    """
    example = _URL_EXAMPLES[component.backend].get(
        component.type, _CALL_ROUTE_HOST + call_path(component.type)
    )
    return (
        f"{backend_variable}={component.backend!r} needs {url_variable}, the full "
        f"route of the endpoint (for example {example}); got {url!r}"
    )
