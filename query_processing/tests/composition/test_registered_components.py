"""A component of a type an extension defines: opened by its class, refused by name.

The composition finds the implementation its backend names, hands it its
block and its resources, and the class builds itself. A backend the component
type does not have is refused before anything opens.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from query_processing.composition.composition_root import CompositionRoot
from query_processing.composition.registered_components import open_registered
from query_processing.composition.registered_names import check_registered_names
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.graph_description import GraphDescription
from rag_components import FAKE_BACKENDS
from shout_components import (
    SHOUT_EXTENSION,
    resources_for,
    shout_graph,
    shouter_settings,
)
from shout_extension.shouting import FakeShouter, ShoutRequest, UpperShouter

SettingsFactory = Callable[..., WorkflowSettings]


async def test_a_registered_component_is_built_from_its_block() -> None:
    shouter = await open_registered(
        "shouter", shouter_settings(suffix="!"), resources_for("shouter")
    )

    assert isinstance(shouter, UpperShouter)
    assert (await shouter.ainvoke(ShoutRequest(text="a"))).text == "A!"


async def test_an_implementation_with_no_settings_is_built_with_none() -> None:
    shouter = await open_registered(
        "shouter", shouter_settings("fake"), resources_for("shouter")
    )

    assert isinstance(shouter, FakeShouter)


async def test_the_composition_opens_a_registered_component_a_stage_calls(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        **FAKE_BACKENDS,
        extensions=[SHOUT_EXTENSION],
        components={"shouter": {"backend": "local", "local": {"suffix": "?"}}},
        graph=GraphDescription.model_validate(shout_graph()),
    )

    async with CompositionRoot(settings) as composition:
        shouter = (await composition.workflow_dependencies()).components()["shouter"]

    assert isinstance(shouter, UpperShouter)
    assert (await shouter.ainvoke(ShoutRequest(text="a"))).text == "A?"


async def test_a_registered_component_handed_in_ready_made_is_used_as_given(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        **FAKE_BACKENDS,
        extensions=[SHOUT_EXTENSION],
        components={"shouter": {"backend": "local"}},
        graph=GraphDescription.model_validate(shout_graph()),
    )
    given = FakeShouter()

    async with CompositionRoot(settings, shouter=given) as composition:
        deps = await composition.workflow_dependencies()

    assert deps.components()["shouter"] is given


def test_a_backend_the_component_type_does_not_have_is_refused_by_name(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        extensions=[SHOUT_EXTENSION], components={"shouter": {"backend": "loud"}}
    )

    with pytest.raises(
        ValueError,
        match=r"QP_COMPONENTS__SHOUTER__BACKEND: unknown shouter backend 'loud' "
        r"\(expected one of: 'fake', 'local'\)",
    ):
        check_registered_names(settings)
