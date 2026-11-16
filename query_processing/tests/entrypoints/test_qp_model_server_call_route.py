"""The model server serves a registered component on its call route, with no code.

The component is one of the test extension's (``shout_extension``): its
request comes in and its answer goes out in its component type's API format,
the one the toolkit builds or the one the type registers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_model_server import (
    create_model_server,
    servable_components,
)
from shout_components import SHOUT_EXTENSION

SettingsFactory = Callable[..., WorkflowSettings]


def model_server(settings: WorkflowSettings, name: str) -> Iterator[TestClient]:
    """The model server of ``name``, its lifespan run."""
    with TestClient(create_model_server(name, CompositionRoot(settings))) as client:
        yield client


def shout_settings(
    settings_factory: SettingsFactory, **components: dict[str, Any]
) -> WorkflowSettings:
    """The fixture's settings, the test extension loaded and ``components`` added."""
    return settings_factory(extensions=[SHOUT_EXTENSION], components=components)


@pytest.fixture
def shouter_app(settings_factory: SettingsFactory) -> Iterator[TestClient]:
    settings = shout_settings(
        settings_factory, shouter={"backend": "local", "local": {"suffix": "!"}}
    )
    yield from model_server(settings, "shouter")


def test_a_registered_component_is_served_on_its_call_route(
    shouter_app: TestClient,
) -> None:
    answer = shouter_app.post("/call/shouter", json={"text": "a"})

    assert answer.status_code == 200
    assert answer.json() == {"text": "A!"}


@pytest.mark.parametrize("body", [{"txt": "a"}, ["a"], {"text": 1}])
def test_a_body_that_is_not_the_request_is_a_422(
    shouter_app: TestClient, body: object
) -> None:
    answer = shouter_app.post("/call/shouter", json=body)

    assert answer.status_code == 422
    assert "ShoutRequest" in answer.json()["detail"]


def test_a_body_that_is_not_json_is_a_422(shouter_app: TestClient) -> None:
    answer = shouter_app.post("/call/shouter", content=b"text=a")

    assert answer.status_code == 422


def test_the_model_probe_answers_for_the_components_name(
    shouter_app: TestClient,
) -> None:
    assert shouter_app.get("/v3/models/shouter").json() == {
        "name": "shouter",
        "role": "shouter",
    }
    assert shouter_app.get("/v3/models/other").status_code == 404
    assert shouter_app.get("/healthz").json() == {"status": "ok", "role": "shouter"}


def test_only_the_call_route_of_its_type_is_served(shouter_app: TestClient) -> None:
    assert shouter_app.post("/v3/embeddings", json={}).status_code == 404
    assert shouter_app.post("/call/whisperer", json={"text": "a"}).status_code == 404


def test_the_api_format_a_type_registers_is_spoken(
    settings_factory: SettingsFactory,
) -> None:
    settings = shout_settings(settings_factory, whisperer={"backend": "local"})
    for app in model_server(settings, "whisperer"):
        answer = app.post("/call/whisperer", json={"words": "QUIET"})

        assert answer.json() == {"words": "quiet"}
        assert app.post("/call/whisperer", json={"text": "a"}).status_code == 422


def test_a_component_the_file_puts_on_served_is_served_in_process(
    settings_factory: SettingsFactory,
) -> None:
    settings = shout_settings(
        settings_factory,
        shouter={
            "backend": "served",
            "served": {"url": "http://shouter-app/call/shouter"},
            "local": {"suffix": "?"},
        },
    )
    for app in model_server(settings, "shouter"):
        assert app.post("/call/shouter", json={"text": "a"}).json() == {"text": "A?"}


def test_a_registered_component_is_servable(settings_factory: SettingsFactory) -> None:
    settings = shout_settings(settings_factory, shouter={"backend": "fake"})

    assert servable_components(settings) == ["embedder", "reranker", "shouter"]
