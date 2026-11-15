"""The workflow app's topology rule: which backends a topology implies.

The deploy tool resolves a topology alias of the deploy settings into each
component's backend (``deploy/topology_alias.py``, ADR 0003). Until the
workflow app drops this rule, it still reads a topology from a closed set of
three, which places the embedder and the reranker: ``local`` in this process,
or ``served`` behind a model server of ours. A component it places follows it
while the component's backend is ``local``; any other backend (an endpoint we
do not run, ``fake``, ``none``) loads no model of ours here and is kept. The
run label that names a configuration is in ``run_label.py``. Pure functions
only: the settings call them at construction.

Example:
    >>> placed_backend("split-reranker", "rerank", "local")
    'served'
    >>> TOPOLOGY_BACKENDS["split-all"].embedding
    'served'
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple

Topology = Literal["monolith", "split-reranker", "split-all"]
# The two components a topology places, by the field of TopologyBackends.
PlacedComponent = Literal["embedding", "rerank"]
# The same two by the name the workflow file gives them, as the settings place
# them (config/rag_settings.py keeps its own copy, _PLACED).
PLACED_BY_NAME: dict[str, PlacedComponent] = {
    "embedder": "embedding",
    "reranker": "rerank",
}
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


def topology_that_moves_nothing(alias: str, backends: Mapping[str, str]) -> Topology:
    """Name the topology to tell the workflow app, so that its rule moves nothing.

    The deploy tool writes every backend a topology alias decides, and the
    workflow app still applies its own rule to ``RAG_TOPOLOGY`` until that rule
    goes. So the app is told the alias when it knows the name and the rule
    leaves ``backends`` as they are, and ``monolith`` otherwise, whose rule
    moves no component.

    Args:
        alias: The topology alias being deployed.
        backends: The backend each component of the workflow file gets, by name.

    Returns:
        ``alias`` when it is a topology whose rule keeps every backend of
        ``backends``, else ``monolith``.

    Example:
        >>> topology_that_moves_nothing("split-all", {"embedder": "served"})
        'split-all'
        >>> topology_that_moves_nothing("split-embedder", {"embedder": "served"})
        'monolith'
    """
    known = next((name for name in TOPOLOGY_BACKENDS if name == alias), None)
    if known is not None and _moves_nothing(known, backends):
        return known
    return "monolith"


def _moves_nothing(topology: Topology, backends: Mapping[str, str]) -> bool:
    placed = {
        name: placed_backend(topology, PLACED_BY_NAME[name], backend)
        for name, backend in backends.items()
        if name in PLACED_BY_NAME
    }
    return all(backends[name] == backend for name, backend in placed.items())
