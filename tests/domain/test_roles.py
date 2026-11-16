"""The Apps a topology needs and the Role each is published as, for every case."""

from __future__ import annotations

from typing import get_args

import pytest

from rag_load_test.domain.roles import (
    AppPart,
    GeneratorServer,
    ModelServerKind,
    Role,
    apps_for_topology,
    generator_role,
    model_servers_for_topology,
    role_for,
)
from rag_load_test.domain.topology import Topology

TOPOLOGIES: tuple[Topology, ...] = get_args(Topology)
# The generator's Role follows from generator_server, not from the topology.
PARTS = tuple(part for part in get_args(AppPart) if part != "generator")
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


def test_a_generator_is_published_after_the_model_servers_and_before_the_workflow() -> (
    None
):
    assert apps_for_topology("monolith", generator=True) == ["generator", "workflow"]
    assert apps_for_topology("split-all", generator=True) == [
        "embedder",
        "reranker",
        "generator",
        "workflow",
    ]


def test_no_generator_unless_asked() -> None:
    assert apps_for_topology("split-all") == ["embedder", "reranker", "workflow"]


def test_the_generators_role_names_its_server() -> None:
    assert generator_role("ovms") == "ovms-generator"
    with pytest.raises(ValueError, match="generator_server"):
        generator_role("none")


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_generators_role_is_not_the_topologys_to_give(topology: Topology) -> None:
    with pytest.raises(ValueError, match="generator_role"):
        role_for(topology, "generator", "ovms")


def test_every_role_is_given_to_some_app() -> None:
    servers: tuple[GeneratorServer, ...] = get_args(GeneratorServer)
    generators = {generator_role(server) for server in servers if server != "none"}
    given = {role for role in EXPECTED_ROLES.values() if role is not None}
    assert given | generators == set(get_args(Role))
