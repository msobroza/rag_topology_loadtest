"""The Topology: where the models run, and so which backends it implies.

A topology is a closed set; it says where the in-process models run, so it
places the embedder and the reranker: ``local`` in this process, or ``api``
behind a model server of ours. A component it places follows it while the
component's backend is ``local``; any other backend (an endpoint we do not
run, ``fake``, ``none``) loads no model of ours here and is kept. The run label
that names a configuration is in ``run_label.py``. Pure functions only: the
settings call them at construction; the Domino rule that ``rag-serve --domino``
applies lives in ``config/domino_rules.py``.

Example:
    >>> placed_backend("split-reranker", "rerank", "local")
    'api'
    >>> TOPOLOGY_BACKENDS["split-all"].embedding
    'api'
"""

from __future__ import annotations

from typing import Literal, NamedTuple

Topology = Literal["monolith", "split-reranker", "split-all"]
# The two components a topology places, by the field of TopologyBackends.
PlacedComponent = Literal["embedding", "rerank"]
LOCAL = "local"


class TopologyBackends(NamedTuple):
    """The embedding and rerank backends one topology implies."""

    embedding: str
    rerank: str


TOPOLOGY_BACKENDS: dict[Topology, TopologyBackends] = {
    "monolith": TopologyBackends(embedding="local", rerank="local"),
    "split-reranker": TopologyBackends(embedding="local", rerank="api"),
    "split-all": TopologyBackends(embedding="api", rerank="api"),
}


def placed_backend(topology: Topology, component: PlacedComponent, stated: str) -> str:
    """Return the backend of a component the topology places.

    The workflow file states ``local`` for a model of ours; the topology moves
    it behind a model server. Any other stated backend loads no model of ours
    into this process, so where ours run does not constrain it.

    Args:
        topology: The topology that implies the backend.
        component: Which component it places: ``embedding`` or ``rerank``.
        stated: The backend the settings give the component.

    Returns:
        The backend the topology implies when ``stated`` is ``local``, else
        ``stated``.

    Example:
        >>> placed_backend("split-all", "embedding", "local")
        'api'
        >>> placed_backend("split-all", "rerank", "none")
        'none'
    """
    if stated != LOCAL:
        return stated
    return getattr(TOPOLOGY_BACKENDS[topology], component)
