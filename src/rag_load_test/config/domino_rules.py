"""The Domino rule: settings made fit for a Domino App (``rag-serve --domino``)."""

from __future__ import annotations

from ..domain.model_api import DOMINO_RUN_TOKEN
from .settings import RagSettings


def apply_domino_rules(settings: RagSettings) -> RagSettings:
    """Make settings fit for a Domino App.

    On Domino a model server is another App, and App-to-App calls need the run's
    token. So for embedding and for rerank, when the backend is ``api`` and the
    key was not set explicitly, the key becomes ``domino``. The generation key is
    never touched: a generation endpoint is not one of our Apps. The input is not
    mutated.

    Args:
        settings: The settings to make fit.

    Returns:
        ``settings`` itself when no key needs a default, else a copy with
        ``embedding_api_key`` and/or ``rerank_api_key`` set to ``domino``.

    Example:
        >>> apply_domino_rules(RagSettings(topology="split-reranker",
        ...     rerank_api_url="http://app/v3/rerank")).rerank_api_key
        'domino'
    """
    update = {
        key: DOMINO_RUN_TOKEN
        for key in _keys_to_default(settings)
        if key not in settings.model_fields_set
    }
    return settings.model_copy(update=update) if update else settings


def _keys_to_default(settings: RagSettings) -> list[str]:
    keys = []
    if settings.embedding_backend == "api":
        keys.append("embedding_api_key")
    if settings.rerank_backend == "api":
        keys.append("rerank_api_key")
    return keys
