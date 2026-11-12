"""The Topology: where the models run, and so which backends it implies.

A topology is a closed set; it says where the in-process models run, so it
decides the embedding and rerank backends. An explicit backend may only repeat
that choice or pick a variant that loads no model of ours into this process:
``api`` (an endpoint called over HTTP), ``fake``, or ``none`` for rerank (the
names are in ``backends.py``; the run label that names a configuration is in
``run_label.py``). Pure functions only: settings call them at construction; the
Domino rule that ``rag-serve --domino`` applies lives in
``config/domino_rules.py``.

Example:
    >>> resolve_rerank_backend("split-reranker", None)
    'api'
    >>> TOPOLOGY_BACKENDS["split-all"].embedding
    'api'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

from .backends import (
    EMBEDDING_VARIANTS,
    RERANK_VARIANTS,
    EmbeddingBackend,
    RerankBackend,
)

Topology = Literal["monolith", "split-reranker", "split-all"]


class TopologyBackends(NamedTuple):
    """The embedding and rerank backends one topology implies."""

    embedding: EmbeddingBackend
    rerank: RerankBackend


TOPOLOGY_BACKENDS: dict[Topology, TopologyBackends] = {
    "monolith": TopologyBackends(embedding="local", rerank="local"),
    "split-reranker": TopologyBackends(embedding="local", rerank="api"),
    "split-all": TopologyBackends(embedding="api", rerank="api"),
}


def resolve_embedding_backend(
    topology: Topology, explicit: EmbeddingBackend | None
) -> EmbeddingBackend:
    """Return the embedding backend for ``topology``.

    Args:
        topology: The topology that implies the backend.
        explicit: The backend asked for, or None for the implied one; it may only
            be the implied backend or one of ``EMBEDDING_VARIANTS``.

    Returns:
        ``explicit`` when set, else the backend the topology implies.

    Raises:
        ValueError: If ``explicit`` contradicts the topology (``local`` where the
            topology implies ``api``); the message names the topologies that
            would allow it.

    Example:
        >>> resolve_embedding_backend("split-all", None)
        'api'
        >>> resolve_embedding_backend("split-all", "local")
        ValueError: RAG_EMBEDDING_BACKEND='local' contradicts ...
    """
    implied = TOPOLOGY_BACKENDS[topology].embedding
    if explicit is None:
        return implied
    allowed = _distinct(implied, *EMBEDDING_VARIANTS)
    _refuse_contradiction(topology, "embedding", explicit, allowed)
    return explicit


def resolve_rerank_backend(
    topology: Topology, explicit: RerankBackend | None
) -> RerankBackend:
    """Return the rerank backend for ``topology``.

    Args:
        topology: The topology that implies the backend.
        explicit: The backend asked for, or None for the implied one; it may only
            be the implied backend or one of ``RERANK_VARIANTS`` (``none`` skips
            reranking).

    Returns:
        ``explicit`` when set, else the backend the topology implies.

    Raises:
        ValueError: If ``explicit`` contradicts the topology (``local`` where the
            topology implies ``api``); the message names the topologies that
            would allow it.

    Example:
        >>> resolve_rerank_backend("monolith", "none")
        'none'
    """
    implied = TOPOLOGY_BACKENDS[topology].rerank
    if explicit is None:
        return implied
    allowed = _distinct(implied, *RERANK_VARIANTS)
    _refuse_contradiction(topology, "rerank", explicit, allowed)
    return explicit


def _refuse_contradiction(
    topology: Topology,
    component: Literal["embedding", "rerank"],
    explicit: str,
    allowed: Sequence[str],
) -> None:
    if explicit in allowed:
        return
    variable = f"RAG_{component.upper()}_BACKEND"
    raise ValueError(
        f"{variable}={explicit!r} contradicts RAG_TOPOLOGY={topology!r}, which allows "
        f"only {_either(allowed)} for {component}; "
        f"{_contradiction_fix(component, explicit, variable)}"
    )


def _contradiction_fix(
    component: Literal["embedding", "rerank"], explicit: str, variable: str
) -> str:
    matching = [
        f"RAG_TOPOLOGY={name}"
        for name, backends in TOPOLOGY_BACKENDS.items()
        if getattr(backends, component) == explicit
    ]
    if not matching:
        return f"drop {variable}"
    return f"set {' or '.join(matching)}, or drop {variable}"


def _distinct(*values: str) -> tuple[str, ...]:
    # The implied backend may also be a variant (``api``): name it once.
    return tuple(dict.fromkeys(values))


def _either(values: Sequence[str]) -> str:
    quoted = [repr(value) for value in values]
    return ", ".join(quoted[:-1]) + f" or {quoted[-1]}"
