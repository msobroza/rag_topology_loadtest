"""The Role each App is published as, and what a model server App can serve."""

from __future__ import annotations

from typing import get_args

import pytest

from rag_load_test.domain.roles import (
    MODEL_SERVER_ROUTES,
    ModelServerKind,
    Role,
    model_server_role,
    workflow_role,
)
from rag_load_test.domain.topology import Topology

KINDS: tuple[ModelServerKind, ...] = get_args(ModelServerKind)
# Every Role, written out: (App, model-server kind or topology) -> Role.
MODEL_SERVER_ROLES = {
    ("embedder", "fastapi"): "fastapi-embedder",
    ("reranker", "fastapi"): "fastapi-reranker",
    ("embedder", "ovms"): "ovms-embedder",
    ("reranker", "ovms"): "ovms-reranker",
}
WORKFLOW_ROLES = {
    "monolith": "monolith",
    "split-reranker": "workflow",
    "split-all": "workflow",
}


@pytest.mark.parametrize(("component", "kind"), list(MODEL_SERVER_ROLES))
def test_the_role_of_each_model_server_app(
    component: str, kind: ModelServerKind
) -> None:
    assert model_server_role(kind, component) == MODEL_SERVER_ROLES[(component, kind)]


@pytest.mark.parametrize("component", ["llm", "reranker_fr", "workflow"])
def test_a_component_no_model_server_app_serves_has_no_role(component: str) -> None:
    with pytest.raises(ValueError, match="no model server App serves") as caught:
        model_server_role("fastapi", component)
    assert str(caught.value) == (
        f"no model server App serves a component named {component!r}: a fastapi "
        "model server serves embedder, reranker"
    )


@pytest.mark.parametrize("topology", get_args(Topology))
def test_the_workflow_role_follows_the_topology_it_is_told(topology: Topology) -> None:
    assert workflow_role(topology) == WORKFLOW_ROLES[topology]


def test_every_role_is_given_to_some_app() -> None:
    given = {*MODEL_SERVER_ROLES.values(), *WORKFLOW_ROLES.values()}
    assert given == set(get_args(Role))


def test_each_kind_serves_an_embedder_and_a_reranker_on_the_v3_routes() -> None:
    expected = {"embedder": "/v3/embeddings", "reranker": "/v3/rerank"}
    assert {kind: MODEL_SERVER_ROUTES[kind] for kind in KINDS} == {
        "fastapi": expected,
        "ovms": expected,
    }
