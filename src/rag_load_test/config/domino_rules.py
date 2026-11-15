"""The Domino rule: settings made fit for a Domino App (``rag-serve --domino``)."""

from __future__ import annotations

from ..domain.api_endpoint import DOMINO_RUN_TOKEN
from .component_settings import ModelComponent
from .rag_settings import RagSettings

# The types a model server of ours serves: an endpoint of theirs may be one of
# our Apps. An LLM's endpoint never is.
_SERVED_TYPES = ("embedder", "reranker")


def apply_domino_rules(settings: RagSettings) -> RagSettings:
    """Make settings fit for a Domino App.

    On Domino a model server is another App, and App-to-App calls need the run's
    token. So for every embedder and reranker, when the backend is ``api`` and
    the key was not set explicitly, the key becomes ``domino``. An LLM's key is
    never touched: a generation endpoint is not one of our Apps. The input is
    not mutated.

    Args:
        settings: The settings to make fit.

    Returns:
        ``settings`` itself when no key needs a default, else a copy whose
        components' ``api.api_key`` is ``domino`` where the rule applies.

    Example:
        >>> apply_domino_rules(RagSettings(topology="split-reranker", components={
        ...     "reranker": {"api": {"url": "http://app/v3/rerank"}}})
        ... ).components["reranker"].api.api_key
        'domino'
    """
    components = {
        name: _with_run_token(component)
        for name, component in settings.components.items()
    }
    if components == settings.components:
        return settings
    return settings.model_copy(update={"components": components})


def _with_run_token(component: ModelComponent) -> ModelComponent:
    if component.type not in _SERVED_TYPES or component.backend != "api":
        return component
    if "api_key" in component.api.model_fields_set:  # set explicitly: kept
        return component
    api = component.api.model_copy(update={"api_key": DOMINO_RUN_TOKEN})
    return component.model_copy(update={"api": api})
