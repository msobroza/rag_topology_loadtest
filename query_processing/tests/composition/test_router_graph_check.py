"""A router's routes reach the start-up checks: its type keys, and the labels they give."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from query_processing.adapters.reranker.base_reranker import BaseReranker
from query_processing.adapters.router.base_router import BaseRouter
from query_processing.application.graph_check import WorkflowGraphError
from query_processing.composition.composition_root import (
    check_workflow_graph,
    component_type_settings,
)
from query_processing.config.routes_file import ROUTES_FILE_VAR
from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import fixture_graph

SettingsFactory = Callable[..., WorkflowSettings]
ROUTER = {"type": "router", "backend": "fake", "embedder": "embedder"}


@pytest.fixture
def routed_settings(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[str, str]], WorkflowSettings]:
    """Settings of the fixture's graph with a router before retrieve, routes a and b."""
    routes = [{"name": n, "description": n, "utterances": [n]} for n in ("a", "b")]
    routes_file = tmp_path / "routes.yaml"
    routes_file.write_text(yaml.safe_dump({"routes": routes}), "utf-8")
    monkeypatch.setenv(ROUTES_FILE_VAR, str(routes_file))

    def make(path_map: dict[str, str]) -> WorkflowSettings:
        graph = fixture_graph()
        graph["stages"]["route"] = {
            "call": "router",
            "reads": ["query_embedding"],
            "writes": ["route"],
        }
        graph["edges"] = [["START", "embed", "route"], ["retrieve", "rerank"]]
        graph["edges"].append(["generate", "END"])
        edge = {"source": "route", "path": "label", "reads": ["route"]}
        graph["conditional_edges"].append({**edge, "path_map": path_map})
        return settings_factory(
            components={"router": {**ROUTER, "routes": ["a", "b"]}}, graph=graph
        )

    return make


def test_the_type_keys_of_a_router_are_its_settings_and_none_for_the_others(
    routed_settings: Callable[[dict[str, str]], WorkflowSettings],
) -> None:
    settings = routed_settings({"a": "retrieve", "b": "END"})

    keys = component_type_settings(settings)

    assert keys["router"] is settings.components["router"]
    assert keys["reranker"] is None


def test_the_graph_check_passes_a_path_map_of_every_route(
    routed_settings: Callable[[dict[str, str]], WorkflowSettings],
) -> None:
    check_workflow_graph(routed_settings({"a": "retrieve", "b": "END"}))


def test_the_graph_check_refuses_a_path_map_that_misses_a_route(
    routed_settings: Callable[[dict[str, str]], WorkflowSettings],
) -> None:
    with pytest.raises(WorkflowGraphError, match=r"path_map lists 'a' \(expected"):
        check_workflow_graph(routed_settings({"a": "retrieve"}))


def test_a_router_answers_with_its_routes_as_the_labels_of_the_route_chosen() -> None:
    class RouterKeys(BaseModel):
        routes: list[str]

    labels = BaseRouter.answer_labels(RouterKeys(routes=["a", "b"]))

    assert labels == {"name": ("a", "b")}


def test_a_router_with_no_routes_in_its_settings_is_refused() -> None:
    with pytest.raises(TypeError, match="a router's settings are None; expected"):
        BaseRouter.answer_labels(None)


def test_another_component_type_declares_no_labels() -> None:
    assert BaseReranker.answer_labels(None) == {}
