"""The Topology: where the models run, and so which backends it implies.

A topology is a closed set; it says where the in-process models run, so it
places the embedder and the reranker: ``local`` in this process, or ``served``
behind a model server of ours. A component it places follows it while the
component's backend is ``local``; any other backend (an endpoint we do not
run, ``fake``, ``none``) loads no model of ours here and is kept. The run label
that names a configuration is in ``run_label.py``. Pure functions only: the
settings call them at construction.

Example:
    >>> placed_backend("split-reranker", "rerank", "local")
    'served'
    >>> TOPOLOGY_BACKENDS["split-all"].embedding
    'served'
"""

from __future__ import annotations

from typing import Literal, NamedTuple

Topology = Literal["monolith", "split-reranker", "split-all"]
# The two components a topology places, by the field of TopologyBackends.
PlacedComponent = Literal["embedding", "rerank"]
LOCAL = "local"
# The backend of a component a model server of ours serves.
SERVED = "served"


class TopologyBackends(NamedTuple):
    """The embedding and rerank backends one topology implies."""

    embedding: str
    rerank: str


TOPOLOGY_BACKENDS: dict[Topology, TopologyBackends] = {
    "monolith": TopologyBackends(embedding=LOCAL, rerank=LOCAL),
    "split-reranker": TopologyBackends(embedding=LOCAL, rerank=SERVED),
    "split-all": TopologyBackends(embedding=SERVED, rerank=SERVED),
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
        'served'
        >>> placed_backend("split-all", "rerank", "none")
        'none'
    """
    if stated != LOCAL:
        return stated
    return getattr(TOPOLOGY_BACKENDS[topology], component)
