"""A registered component on ``served``: the class the toolkit builds calls a model server.

The model server runs in process, behind ``httpx.ASGITransport``, and a test
that needs a server of another shape answers through ``httpx.MockTransport``:
no socket is opened.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.composition.registered_components import served_class_of
from query_processing.composition.registered_names import check_registered_names
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.components.base_component import BaseComponent
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.graph_description import GraphDescription
from query_processing.domain.registry import Registry
from query_processing.entrypoints.qp_model_server import create_model_server
from rag_components import FAKE_BACKENDS
from shout_components import SHOUT_EXTENSION, shout_graph
from shout_extension.shouting import Shouted, ShoutRequest
from shout_extension.whispering import (
    SERVED_MARK,
    BaseMurmurer,
    BaseWhisperer,
    FakeMurmurer,
    LowerWhisperer,
    MarkingServedWhisperer,
)

SettingsFactory = Callable[..., WorkflowSettings]
SHOUTER_URL = "http://shouter-app/call/shouter"
WHISPERER_URL = "http://whisperer-app/call/whisperer"


def served_settings(
    settings_factory: SettingsFactory, name: str, url: str, **entry: Any
) -> WorkflowSettings:
    """The fixture's fake settings, ``name`` on ``served`` at ``url`` and called last."""
    component = {"backend": "served", "served": {"url": url}, **entry}
    graph = shout_graph()
    graph["stages"]["shout"]["call"] = name
    return settings_factory(
        **FAKE_BACKENDS,
        extensions=[SHOUT_EXTENSION],
        components={name: component},
        graph=GraphDescription.model_validate(graph),
    )


async def call(
    settings: WorkflowSettings,
    name: str,
    transport: httpx.AsyncBaseTransport,
    text: str = "a",
) -> tuple[object, Shouted]:
    """Open ``name`` as the workflow app does, call it once; return it and its answer."""
    async with CompositionRoot(settings, transport=transport) as composition:
        component = (await composition.workflow_dependencies()).components()[name]
        return component, await component.ainvoke(ShoutRequest(text=text))


def answering(
    body: object, sent: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    """A model server that answers every call with ``body``, recording each."""

    def answer(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(request)
        return httpx.Response(200, json=body)

    return httpx.MockTransport(answer)


async def test_the_served_class_calls_the_model_server_and_reads_the_answer(
    settings_factory: SettingsFactory,
) -> None:
    settings = served_settings(
        settings_factory, "shouter", SHOUTER_URL, local={"suffix": "!"}
    )
    server = create_model_server("shouter", CompositionRoot(settings))

    with TestClient(server):  # runs its lifespan, which opens the shouter
        transport = httpx.ASGITransport(app=server)
        served, answer = await call(settings, "shouter", transport)

    assert type(served).__name__ == "ServedShouter"
    assert answer == Shouted(text="A!")


async def test_the_request_goes_on_the_wire_in_the_api_format(
    settings_factory: SettingsFactory,
) -> None:
    sent: list[httpx.Request] = []
    settings = served_settings(settings_factory, "shouter", SHOUTER_URL)

    await call(settings, "shouter", answering({"text": "A"}, sent))

    assert [str(request.url) for request in sent] == [SHOUTER_URL]
    assert sent[0].read() == b'{"text":"a"}'


async def test_an_answer_of_another_shape_is_a_dependency_error(
    settings_factory: SettingsFactory,
) -> None:
    settings = served_settings(settings_factory, "shouter", SHOUTER_URL)

    with pytest.raises(RagDependencyError) as failed:
        await call(settings, "shouter", answering({"shout": "A"}))

    assert failed.value.component == "shouter"
    assert failed.value.target == SHOUTER_URL
    assert "malformed response" in failed.value.detail


@pytest.mark.parametrize(("retries", "attempts"), [(0, 1), (2, 3)])
async def test_the_served_class_is_called_with_the_components_http_settings(
    settings_factory: SettingsFactory, retries: int, attempts: int
) -> None:
    sent: list[httpx.Request] = []

    def busy(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(503)

    http = {"retries": retries, "backoff_s": 0}
    settings = served_settings(settings_factory, "shouter", SHOUTER_URL, http=http)

    with pytest.raises(RagDependencyError):
        await call(settings, "shouter", httpx.MockTransport(busy))

    assert len(sent) == attempts


async def test_a_served_class_the_type_registers_takes_the_built_ones_place(
    settings_factory: SettingsFactory,
) -> None:
    settings = served_settings(settings_factory, "whisperer", WHISPERER_URL)
    server = create_model_server("whisperer", CompositionRoot(settings))

    with TestClient(server):
        served, answer = await call(
            settings, "whisperer", httpx.ASGITransport(app=server), "QUIET"
        )

    assert BaseWhisperer.backends == {
        "local": LowerWhisperer,
        "served": MarkingServedWhisperer,
    }
    assert isinstance(served, MarkingServedWhisperer)
    assert answer == Shouted(text="quiet" + SERVED_MARK)


def test_served_is_a_backend_of_every_registered_type(
    settings_factory: SettingsFactory,
) -> None:
    check_registered_names(served_settings(settings_factory, "shouter", SHOUTER_URL))


def test_a_served_component_with_no_url_is_refused_naming_the_variable(
    settings_factory: SettingsFactory,
) -> None:
    # Refused when the settings are read, before anything opens.
    with pytest.raises(ValueError, match="QP_COMPONENTS__SHOUTER__SERVED__URL"):
        served_settings(settings_factory, "shouter", "")


def test_a_served_class_that_is_no_served_call_mixin_is_refused_naming_it() -> None:
    types: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")

    class BaseEcho(
        BaseComponent[ShoutRequest, Shouted], component_type="echo", registry=types
    ):
        """Gives the text back."""

    class UnmixedServedEcho(BaseEcho, backend="served"):
        def invoke(self, request: ShoutRequest) -> Shouted:
            return Shouted(text=request.text)

    assert BaseEcho.backends == {"served": UnmixedServedEcho}
    with pytest.raises(TypeError, match=r"UnmixedServedEcho .* ServedCallMixin"):
        served_class_of(BaseEcho)


def test_a_model_server_of_a_type_with_no_local_backend_names_the_variable(
    settings_factory: SettingsFactory,
) -> None:
    settings = served_settings(
        settings_factory, "murmurer", "http://murmurer-app/call/murmurer"
    )
    server = create_model_server("murmurer", CompositionRoot(settings))

    assert BaseMurmurer.backends == {"fake": FakeMurmurer}  # no local one
    with (
        pytest.raises(ValueError, match="QP_COMPONENTS__MURMURER__BACKEND='served'"),
        TestClient(server),
    ):
        pass
