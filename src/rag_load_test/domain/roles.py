"""The Apps a Topology needs, and the Role each one is published as.

Both follow from ``TOPOLOGY_BACKENDS``: a component whose implied backend is
``api`` is served by a model server App of ours, and the workflow App is always
there, last, since it is published with the model servers' URLs. The Role is
what ``domino/app.sh`` reads from ``RAG_ROLE`` to start an App: ``role_for``
gives it for a Topology, an App part and a model-server kind. The launch script
stays hand-written; a test runs every script the deploy tool writes through it.

Example:
    >>> apps_for_topology("split-reranker")
    ['reranker', 'workflow']
    >>> role_for("split-all", "embedder", "ovms")
    'ovms-embedder'
"""

from __future__ import annotations

from typing import Literal

from .topology import TOPOLOGY_BACKENDS, Topology

# What one Domino App is published as: RAG_ROLE, read by domino/app.sh.
Role = Literal[
    "monolith",
    "workflow",
    "ovms-reranker",
    "ovms-embedder",
    "fastapi-reranker",
    "fastapi-embedder",
]
# Which App of a Topology it is: the workflow app, or a model server.
AppPart = Literal["embedder", "reranker", "workflow"]
ModelServerPart = Literal["embedder", "reranker"]
# What serves a model server App: our FastAPI model server, or OVMS.
ModelServerKind = Literal["fastapi", "ovms"]

_MODEL_SERVER_ROLES: dict[tuple[ModelServerKind, ModelServerPart], Role] = {
    ("fastapi", "embedder"): "fastapi-embedder",
    ("fastapi", "reranker"): "fastapi-reranker",
    ("ovms", "embedder"): "ovms-embedder",
    ("ovms", "reranker"): "ovms-reranker",
}


def model_servers_for_topology(topology: Topology) -> list[ModelServerPart]:
    """List the model server Apps a topology needs, in deploy order.

    Args:
        topology: The topology being deployed.

    Returns:
        ``embedder`` when the topology implies the ``api`` embedding backend,
        then ``reranker`` when it implies the ``api`` rerank backend.

    Example:
        >>> model_servers_for_topology("split-all")
        ['embedder', 'reranker']
    """
    implied = TOPOLOGY_BACKENDS[topology]
    served: tuple[tuple[ModelServerPart, str], ...] = (
        ("embedder", implied.embedding),
        ("reranker", implied.rerank),
    )
    return [part for part, backend in served if backend == "api"]


def apps_for_topology(topology: Topology) -> list[AppPart]:
    """List every App a topology needs, in deploy order: the workflow App last.

    Args:
        topology: The topology being deployed.

    Returns:
        Its model server Apps, then ``workflow``.

    Example:
        >>> apps_for_topology("monolith")
        ['workflow']
    """
    return [*model_servers_for_topology(topology), "workflow"]


def role_for(topology: Topology, part: AppPart, kind: ModelServerKind) -> Role:
    """Return the Role an App of a topology is published as.

    Args:
        topology: The topology being deployed.
        part: Which of its Apps.
        kind: What serves its model server Apps; the workflow App's Role does not
            depend on it.

    Returns:
        ``monolith`` for the workflow App of a topology with no model server,
        ``workflow`` for that of a split topology, else ``<kind>-<part>``.

    Raises:
        ValueError: If the topology has no such App.

    Example:
        >>> role_for("monolith", "workflow", "fastapi")
        'monolith'
    """
    servers = model_servers_for_topology(topology)
    if part == "workflow":
        return "workflow" if servers else "monolith"
    if part not in servers:
        apps = ", ".join(apps_for_topology(topology))
        raise ValueError(
            f"the {topology} topology has no {part} App; its Apps are: {apps}"
        )
    return _MODEL_SERVER_ROLES[(kind, part)]
