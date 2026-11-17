"""``HttpResources``: how patiently a component on ``served`` or ``api`` is called."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from query_processing.composition.http_resources import HttpResources
from query_processing.composition.resource_scope import ResourceScope
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.infrastructure.http.http_policy import HttpPolicy

SettingsFactory = Callable[..., WorkflowSettings]


@pytest.mark.parametrize(
    ("own", "policy"),
    [
        ({}, HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.2)),
        ({"timeout_s": 120.0}, HttpPolicy(timeout_s=120.0, retries=2, backoff_s=0.2)),
        (
            {"retries": 0, "backoff_s": 1.5},
            HttpPolicy(timeout_s=30.0, retries=0, backoff_s=1.5),
        ),
    ],
    ids=["none-set", "its-timeout", "its-retries-and-backoff"],
)
def test_a_component_has_its_own_values_and_the_default_for_the_rest(
    settings_factory: SettingsFactory, own: dict[str, float], policy: HttpPolicy
) -> None:
    settings = settings_factory(components__reranker__http=own)
    http = HttpResources(settings, ResourceScope("test"))

    assert http.policy_of(settings.components["reranker"]) == policy


def test_the_default_is_set_once_for_every_component(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(http={"timeout_s": 5.0, "retries": 0, "backoff_s": 0})
    http = HttpResources(settings, ResourceScope("test"))

    models = settings.model_components().items()
    policies = {name: http.policy_of(c) for name, c in models}

    assert policies == {
        "embedder": HttpPolicy(timeout_s=5.0, retries=0, backoff_s=0.0),
        "reranker": HttpPolicy(timeout_s=5.0, retries=0, backoff_s=0.0),
        # The tests' file gives the LLM its own timeout: an answer is slow.
        "llm": HttpPolicy(timeout_s=120.0, retries=0, backoff_s=0.0),
    }
