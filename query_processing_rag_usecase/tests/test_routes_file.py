"""The RAG's ``routes.yaml``: its two routes and their utterances, as a router reads them."""

from __future__ import annotations

from pathlib import Path

import pytest
from query_processing.extension_api import RouterSettings, WorkflowSettings

from router_workflow import settings_on_fakes


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkflowSettings:
    return settings_on_fakes(tmp_path, monkeypatch)


def test_each_route_of_the_router_has_five_to_ten_utterances(
    settings: WorkflowSettings,
) -> None:
    routes = {route.name: route for route in settings.routes}
    router = settings.components["router"]
    assert isinstance(router, RouterSettings)
    for name in router.routes:
        assert 5 <= len(routes[name].utterances) <= 10


def test_the_file_holds_the_two_routes_of_the_design(
    settings: WorkflowSettings,
) -> None:
    routes = {route.name: route for route in settings.routes}

    assert list(routes) == ["self_contained", "leans_on_history"]
    assert [len(route.utterances) for route in routes.values()] == [8, 8]
    assert routes["self_contained"].utterances[0] == (
        "Quelle est la franchise habituelle en cas de collision ?"
    )
    assert routes["leans_on_history"].utterances[-1] == "Et sinon, que se passe-t-il ?"


def test_no_utterance_is_listed_twice(settings: WorkflowSettings) -> None:
    utterances = [u for route in settings.routes for u in route.utterances]

    assert len(set(utterances)) == len(utterances) == 16
