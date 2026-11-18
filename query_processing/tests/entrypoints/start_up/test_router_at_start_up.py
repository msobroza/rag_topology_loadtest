"""A router's routes are the labels of the field it writes, checked when ``qp-serve`` starts.

Each case writes the fixture's graph with a router between embed and retrieve,
its routes in ``routes.yaml`` in the working directory, and a label edge on the
route it chose. A ``path_map`` that differs from the router's routes is a usage
error before anything opens; one that has a row for each starts and answers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES, shell_settings
from rag_graphs import fixture_components, fixture_graph, write_workflow_file

QUESTION = "How many paid vacation days do employees accrue each year?"
Graph = dict[str, Any]


@pytest.fixture
def served_apps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """All-fake backends, no QP_* from the shell, an empty directory, no real server.

    Returns:
        The apps ``qp-serve`` asked uvicorn to serve.
    """
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    apps: list[Any] = []
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda app, **kw: apps.append(app))
    return apps


def routed_graph(path_map: dict[str, str]) -> Graph:
    """The fixture's graph, a router choosing a route between embed and retrieve."""
    graph = fixture_graph()
    graph["stages"]["choose_route"] = {
        "call": "router",
        "reads": ["query_embedding"],
        "writes": ["route"],
    }
    graph["edges"][0] = ["START", "embed", "choose_route"]
    graph["edges"].append(["retrieve", "rerank"])
    graph["conditional_edges"].append(
        {
            "source": "choose_route",
            "path": "label",
            "reads": ["route"],
            "path_map": path_map,
        }
    )
    return graph


def start_routed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    router_routes: list[str],
    path_map: dict[str, str],
) -> int:
    """Run ``qp-serve`` on a routed graph, its router listing ``router_routes``."""
    catalogue = [
        {"name": name, "description": name, "utterances": [name]}
        for name in router_routes
    ]
    routes_file = tmp_path / "routes.yaml"  # in the working directory
    routes_file.write_text(yaml.safe_dump({"routes": catalogue}), "utf-8")
    router = {"type": "router", "backend": "fake", "embedder": "embedder"}
    components = {**fixture_components(), "router": {**router, "routes": router_routes}}
    workflow = write_workflow_file(
        tmp_path / "workflow.yaml", routed_graph(path_map), components
    )
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))
    return qp_serve.main([])


def start_up_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    **routed: Any,
) -> str:
    """Run ``start_routed``, which must stop with a usage error; return what it said."""
    with pytest.raises(SystemExit) as exited:
        start_routed(tmp_path, monkeypatch, **routed)
    assert exited.value.code == 2
    return capsys.readouterr().err


def test_a_path_map_that_misses_a_route_is_refused_at_start_up(
    served_apps: list[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refusal = start_up_refusal(
        tmp_path, monkeypatch, capsys, router_routes=["a", "b"], path_map={"a": "embed"}
    )

    assert "path_map lists 'a'" in refusal
    assert "label 'route' takes, and no other: 'a', 'b'" in refusal
    assert "conditional edge from 'choose_route' on 'label'" in refusal
    assert served_apps == []  # the server never started, so nothing opened


def test_a_path_map_with_a_route_the_router_lacks_is_refused_at_start_up(
    served_apps: list[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_map = {"a": "retrieve", "b": "END", "c": "END"}
    refusal = start_up_refusal(
        tmp_path, monkeypatch, capsys, router_routes=["a", "b"], path_map=path_map
    )

    assert "path_map lists 'a', 'b', 'c' (expected one row for each label" in refusal
    assert served_apps == []


def test_a_path_map_of_every_route_starts_and_takes_the_route_chosen(
    served_apps: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fake router chooses its first route, a, which goes on to retrieve;
    # b would end the run with no answer.
    path_map = {"a": "retrieve", "b": "END"}

    assert start_routed(tmp_path, monkeypatch, ["a", "b"], path_map) == 0

    [app] = served_apps
    with TestClient(app) as client:
        response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    assert response.json()["answer"]
