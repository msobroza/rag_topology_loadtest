"""A component's settings class is found by its type, in ``COMPONENT_TYPE_SETTINGS``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from pydantic import ValidationError

from query_processing.config import component_settings
from query_processing.config.component_settings import (
    COMPONENT_TYPE_SETTINGS,
    ComponentSettings,
    EmbedderSettings,
    LlmSettings,
    RerankerSettings,
    VectorStoreSettings,
)
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.registry import Registry

SettingsFactory = Callable[..., WorkflowSettings]


class CounterSettings(ComponentSettings):
    """A component type of the tests' own, with one key of its own."""

    type: Literal["counter"] = "counter"
    start: int = 0


@pytest.fixture
def with_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The four types of the toolkit, and ``counter`` registered beside them."""
    registry: Registry[type[ComponentSettings]] = Registry("component type")
    for name in COMPONENT_TYPE_SETTINGS.names():
        registry.register(name)(COMPONENT_TYPE_SETTINGS.get(name))
    registry.register("counter")(CounterSettings)
    monkeypatch.setattr(component_settings, "COMPONENT_TYPE_SETTINGS", registry)


def test_the_four_types_of_the_toolkit_are_registered() -> None:
    assert {
        n: COMPONENT_TYPE_SETTINGS.get(n) for n in COMPONENT_TYPE_SETTINGS.names()
    } == {
        "embedder": EmbedderSettings,
        "llm": LlmSettings,
        "reranker": RerankerSettings,
        "vector_store": VectorStoreSettings,
    }


@pytest.mark.usefixtures("with_counter")
def test_a_registered_component_type_is_read_with_its_settings_class(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(components={"counter": {"backend": "fake", "start": 3}})

    counter = settings.components["counter"]
    assert counter == CounterSettings(backend="fake", start=3)


@pytest.mark.usefixtures("with_counter")
def test_an_unknown_key_of_a_registered_type_is_refused(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(
        ValidationError, match=r"'counter' component type has no 'strat'"
    ):
        settings_factory(components={"counter": {"backend": "fake", "strat": 3}})


def test_an_unknown_type_names_the_type_found_and_lists_the_known_ones(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError) as refused:
        settings_factory(components={"gen": {"type": "generator", "backend": "fake"}})

    message = str(refused.value)
    assert "components.gen" in message
    assert (
        "unknown component type 'generator' (expected one of: "
        "'embedder', 'llm', 'reranker', 'vector_store')"
    ) in message


def test_an_entry_that_is_no_mapping_is_refused(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(
        ValidationError, match=r"components\.llm\n.*is 'fake'; expected"
    ):
        settings_factory(components={"llm": "fake"})


def test_a_dump_of_the_settings_keeps_each_types_own_keys(
    settings_factory: SettingsFactory,
) -> None:
    dumped = settings_factory().model_dump()["components"]

    assert dumped["embedder"]["query_prefix"] == "query: "
    assert dumped["vector_store"]["fake"] == {"seed_documents": 200}
