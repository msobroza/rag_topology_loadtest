"""Where the routes file is found, what it feeds, and the routes a router may list."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from query_processing.config.router_settings import RouterSettings
from query_processing.config.routes_file import ROUTES_FILE_VAR, routes_file
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.routes import Route
from rag_graphs import fixture_graph, write_workflow_file

SettingsFactory = Callable[..., WorkflowSettings]


def route(name: str) -> dict[str, Any]:
    """A route of the catalogue: its name, a description and one utterance."""
    return {"name": name, "description": f"about {name}", "utterances": [f"{name}?"]}


def write_routes(path: Path, names: list[str]) -> Path:
    """Write a routes file of the routes ``names``; return it."""
    path.write_text(yaml.safe_dump({"routes": [route(n) for n in names]}), "utf-8")
    return path


def router(routes: list[str], backend: str = "fake") -> dict[str, Any]:
    """A router component's entry, which reads the fixture's embedder."""
    return {
        "type": "router",
        "backend": backend,
        "embedder": "embedder",
        "routes": routes,
    }


@pytest.fixture
def here(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty working directory, with ``QP_ROUTES_FILE`` unset."""
    monkeypatch.delenv(ROUTES_FILE_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def settings_with_router(
    tmp_path: Path, routes_in_file: list[str], routes_of_router: list[str]
) -> WorkflowSettings:
    """Settings whose routes file lists ``routes_in_file``, and one router."""
    write_routes(tmp_path / "routes.yaml", routes_in_file)
    return WorkflowSettings(
        _env_file=None,  # type: ignore[call-arg]
        components={"router": router(routes_of_router)},
    )


@pytest.mark.usefixtures("here")
def test_a_router_that_lists_a_route_the_file_lacks_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"route 'c'.*routes.yaml.*'a', 'b'"):
        settings_with_router(
            tmp_path, routes_in_file=["a", "b"], routes_of_router=["a", "c"]
        )


@pytest.mark.usefixtures("here")
def test_the_routes_file_here_feeds_the_routes_and_a_router_reads_its_keys(
    tmp_path: Path,
) -> None:
    settings = settings_with_router(tmp_path, ["a", "b", "c"], ["c", "a"])

    assert routes_file() == Path("routes.yaml")
    assert settings.routes[0] == Route(
        name="a", description="about a", utterances=["a?"]
    )
    component = settings.components["router"]
    assert isinstance(component, RouterSettings)
    assert (component.embedder, component.routes) == ("embedder", ["c", "a"])


def test_the_named_routes_file_wins_over_the_working_directory(
    here: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_routes(here / "routes.yaml", ["here"])
    named = write_routes(tmp_path / "named.yaml", ["named", "other"])
    monkeypatch.setenv(ROUTES_FILE_VAR, str(named))

    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]

    assert [r.name for r in settings.routes] == ["named", "other"]


@pytest.mark.usefixtures("here")
def test_no_file_is_needed_when_no_router_is_declared(
    settings_factory: SettingsFactory,
) -> None:
    assert routes_file() is None
    assert settings_factory().routes == []


@pytest.mark.usefixtures("here")
def test_a_router_with_no_routes_file_is_refused(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match=r"route 'a'.*none found.*none\)"):
        settings_factory(components={"router": router(["a", "b"])})


@pytest.mark.usefixtures("here")
def test_a_router_lists_two_routes_at_least(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match=r"routes\n.*at least 2 items"):
        settings_with_router(tmp_path, ["a", "b"], ["a"])


@pytest.mark.usefixtures("here")
def test_a_route_named_twice_in_the_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="names the route 'a' twice"):
        settings_with_router(tmp_path, ["a", "b", "a"], ["a", "b"])


@pytest.mark.parametrize(
    "content", ["- a\n- b\n", "routes: a\n", "routes: [\n"], ids=["list", "str", "yaml"]
)
def test_a_routes_file_of_another_shape_is_refused_naming_it(
    here: Path, content: str
) -> None:
    (here / "routes.yaml").write_text(content, encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"routes file \(QP_ROUTES_FILE.*'routes.yaml'"
    ):
        WorkflowSettings(_env_file=None)  # type: ignore[call-arg]


def test_a_workflow_file_with_a_routes_key_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = write_workflow_file(tmp_path / "workflow.yaml", fixture_graph())
    blocks = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    workflow.write_text(yaml.safe_dump({**blocks, "routes": [route("a")]}), "utf-8")
    monkeypatch.setenv("QP_WORKFLOW_FILE", str(workflow))

    with pytest.raises(
        ValueError, match=r"has a 'routes' key: the routes are written in routes\.yaml"
    ):
        WorkflowSettings(_env_file=None)  # type: ignore[call-arg]
