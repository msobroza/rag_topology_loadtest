"""``HttpResources``: how patiently a component is called, and the client's pool."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
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


class RecordingClientFactory:
    """Builds a real HTTP/1.1 client, keeping the keywords it was asked for."""

    def __init__(self) -> None:
        self.keywords: dict[str, Any] = {}

    def __call__(self, **keywords: Any) -> httpx.AsyncClient:
        self.keywords = keywords
        # h2 is not installed here: httpx would refuse http2=True on building.
        return httpx.AsyncClient(**{**keywords, "http2": False})


async def client_keywords(
    settings: WorkflowSettings, *, h2_installed: bool = False
) -> dict[str, Any]:
    """The keywords the shared client is built with under these settings."""
    factory = RecordingClientFactory()
    async with ResourceScope("test") as scope:
        http = HttpResources(
            settings,
            scope,
            client_factory=factory,
            h2_installed=lambda: h2_installed,
        )
        await http.client()
    return factory.keywords


async def test_the_client_at_the_defaults_is_built_with_httpx_own_pool(
    settings_factory: SettingsFactory,
) -> None:
    keywords = await client_keywords(settings_factory())

    # httpx 0.28.1's DEFAULT_LIMITS: the client built before the pool settings.
    assert keywords["limits"] == httpx.Limits(
        max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0
    )
    assert (keywords["http2"], keywords["timeout"]) == (False, 30.0)


async def test_the_client_is_built_with_the_blocks_limits(
    settings_factory: SettingsFactory,
) -> None:
    pool = {"max_connections": 8, "max_keepalive_connections": 4}
    settings = settings_factory(http={**pool, "keepalive_expiry_s": 1.5})

    keywords = await client_keywords(settings)

    assert keywords["limits"] == httpx.Limits(
        max_connections=8, max_keepalive_connections=4, keepalive_expiry=1.5
    )


async def test_http2_with_h2_installed_reaches_the_client(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(http={"http2": True})

    keywords = await client_keywords(settings, h2_installed=True)

    assert keywords["http2"] is True


def test_http2_without_h2_is_refused_naming_the_extra(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(http={"http2": True})

    with pytest.raises(ValueError, match=r"QP_HTTP__HTTP2=true.*httpx\[http2\]"):
        HttpResources(settings, ResourceScope("test"), h2_installed=lambda: False)
