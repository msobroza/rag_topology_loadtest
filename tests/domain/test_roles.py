"""The Apps a topology needs and the Role each is published as, for every case."""

from __future__ import annotations

from typing import get_args

import pytest

from rag_load_test.domain.roles import (
    AppPart,
    ModelServerKind,
    Role,
    apps_for_topology,
    model_servers_for_topology,
    role_for,
)
from rag_load_test.domain.topology import Topology

TOPOLOGIES: tuple[Topology, ...] = get_args(Topology)
PARTS: tuple[AppPart, ...] = get_args(AppPart)
KINDS: tuple[ModelServerKind, ...] = get_args(ModelServerKind)
# Every Role, written out: (topology, App part, model-server kind) -> Role, or
# None when the topology has no such App.
EXPECTED_ROLES: dict[tuple[str, str, str], str | None] = {
    ("monolith", "workflow", "fastapi"): "monolith",
    ("monolith", "workflow", "ovms"): "monolith",
    ("monolith", "embedder", "fastapi"): None,
    ("monolith", "embedder", "ovms"): None,
    ("monolith", "reranker", "fastapi"): None,
    ("monolith", "reranker", "ovms"): None,
    ("split-reranker", "workflow", "fastapi"): "workflow",
    ("split-reranker", "workflow", "ovms"): "workflow",
    ("split-reranker", "embedder", "fastapi"): None,
    ("split-reranker", "embedder", "ovms"): None,
    ("split-reranker", "reranker", "fastapi"): "fastapi-reranker",
    ("split-reranker", "reranker", "ovms"): "ovms-reranker",
    ("split-all", "workflow", "fastapi"): "workflow",
    ("split-all", "workflow", "ovms"): "workflow",
    ("split-all", "embedder", "fastapi"): "fastapi-embedder",
    ("split-all", "embedder", "ovms"): "ovms-embedder",
    ("split-all", "reranker", "fastapi"): "fastapi-reranker",
    ("split-all", "reranker", "ovms"): "ovms-reranker",
}


@pytest.mark.parametrize(
    ("topology", "apps"),
    [
        ("split-all", ["embedder", "reranker", "workflow"]),
        ("split-reranker", ["reranker", "workflow"]),
        ("monolith", ["workflow"]),
    ],
)
def test_each_topology_needs_its_apps_in_deploy_order(
    topology: Topology, apps: list[str]
) -> None:
    assert apps_for_topology(topology) == apps
    assert model_servers_for_topology(topology) == apps[:-1]


def test_the_expected_table_covers_every_combination() -> None:
    every = {(t, p, k) for t in TOPOLOGIES for p in PARTS for k in KINDS}
    assert set(EXPECTED_ROLES) == every


@pytest.mark.parametrize(
    ("topology", "part", "kind"),
    [key for key, role in EXPECTED_ROLES.items() if role is not None],
)
def test_the_role_of_each_app(
    topology: Topology, part: AppPart, kind: ModelServerKind
) -> None:
    assert role_for(topology, part, kind) == EXPECTED_ROLES[(topology, part, kind)]


@pytest.mark.parametrize(
    ("topology", "part", "kind"),
    [key for key, role in EXPECTED_ROLES.items() if role is None],
)
def test_an_app_the_topology_lacks_has_no_role(
    topology: Topology, part: AppPart, kind: ModelServerKind
) -> None:
    with pytest.raises(ValueError, match=f"the {topology} topology has no {part} App"):
        role_for(topology, part, kind)


def test_every_role_is_given_to_some_app() -> None:
    given = {role for role in EXPECTED_ROLES.values() if role is not None}
    assert given == set(get_args(Role))
