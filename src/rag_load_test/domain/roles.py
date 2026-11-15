"""The Role each App is published as, and what a model server App can serve.

Which Apps a topology needs is the deploy tool's to say: one model server App
per component its topology alias serves, then the workflow App
(``deploy/topology_alias.py``). The Role is what ``domino/app.sh`` reads from
``RAG_ROLE`` to start an App. The launch script stays hand-written; a test runs
every script the deploy tool writes through it.

Example:
    >>> model_server_role("ovms", "embedder")
    'ovms-embedder'
    >>> workflow_role("split-reranker")
    'workflow'
"""

from __future__ import annotations

from typing import Literal

from .api_endpoint import MODEL_SERVER_EMBEDDINGS_PATH, MODEL_SERVER_RERANK_PATH
from .topology import SERVED, TOPOLOGY_BACKENDS, Topology

# What one Domino App is published as: RAG_ROLE, read by domino/app.sh.
Role = Literal[
    "monolith",
    "workflow",
    "ovms-reranker",
    "ovms-embedder",
    "fastapi-reranker",
    "fastapi-embedder",
]
# Which App of a topology it is: the workflow App, or the model server of one
# served component, named after that component.
AppPart = str
WORKFLOW_APP = "workflow"
# A component a model server App serves: app.sh starts it by a Role named after
# its type, so the component is the one named after its type.
ModelServerPart = Literal["embedder", "reranker"]
# What serves a model server App: our FastAPI model server, or OVMS.
ModelServerKind = Literal["fastapi", "ovms"]

# What each model-server kind can serve, by component type, and the /v3 route it
# serves it on: both serve an embedder and a reranker on OVMS's routes. The one
# place that says so, until the model-server kinds become a registry, each kind
# holding its own.
_V3_ROUTES: dict[str, str] = {
    "embedder": MODEL_SERVER_EMBEDDINGS_PATH,
    "reranker": MODEL_SERVER_RERANK_PATH,
}
MODEL_SERVER_ROUTES: dict[ModelServerKind, dict[str, str]] = {
    "fastapi": _V3_ROUTES,
    "ovms": _V3_ROUTES,
}
_MODEL_SERVER_ROLES: dict[tuple[str, str], Role] = {
    ("fastapi", "embedder"): "fastapi-embedder",
    ("fastapi", "reranker"): "fastapi-reranker",
    ("ovms", "embedder"): "ovms-embedder",
    ("ovms", "reranker"): "ovms-reranker",
}


def model_server_role(kind: ModelServerKind, component: str) -> Role:
    """Return the Role of the model server App that serves ``component``.

    Args:
        kind: What serves it: ``fastapi`` or ``ovms``.
        component: The served component's name, which is its type.

    Returns:
        ``<kind>-<component>``.

    Raises:
        ValueError: If no model server App serves a component of that name.

    Example:
        >>> model_server_role("fastapi", "reranker")
        'fastapi-reranker'
    """
    role = _MODEL_SERVER_ROLES.get((kind, component))
    if role is None:
        raise ValueError(
            f"no model server App serves a component named {component!r}: a "
            f"{kind} model server serves {', '.join(MODEL_SERVER_ROUTES[kind])}"
        )
    return role


def workflow_role(topology: Topology) -> Role:
    """Return the Role of the workflow App the workflow app is told ``topology``.

    ``domino/app.sh`` pairs each workflow Role with the topologies it serves:
    ``monolith`` with ``monolith``, ``workflow`` with the two split topologies.

    Args:
        topology: The ``RAG_TOPOLOGY`` the workflow App is given.

    Returns:
        ``workflow`` when the topology serves a component, else ``monolith``.

    Example:
        >>> workflow_role("monolith")
        'monolith'
    """
    if SERVED in TOPOLOGY_BACKENDS[topology]:
        return "workflow"
    return "monolith"
