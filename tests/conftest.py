"""Shared fixtures. No model weights, no network, no Elasticsearch anywhere in this suite."""

from __future__ import annotations

import os

# locust gevent-patches the interpreter at import unless this is set; it must run
# before any ``import locust`` (the locustfile tests import it).
os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

from collections.abc import AsyncIterator, Callable  # noqa: E402

import pytest  # noqa: E402

from rag_load_test.adapters.rag_ports import RagPorts  # noqa: E402
from rag_load_test.config.settings import RagSettings  # noqa: E402
from rag_load_test.domain.ports import RagDependencies  # noqa: E402


@pytest.fixture
def settings_factory() -> Callable[..., RagSettings]:
    """RagSettings without reading a stray ``.env``; keyword overrides allowed."""

    def make(**overrides: object) -> RagSettings:
        return RagSettings(_env_file=None, **overrides)  # type: ignore[call-arg]

    return make


@pytest.fixture
def fake_settings(settings_factory: Callable[..., RagSettings]) -> RagSettings:
    """Every backend set to ``fake``: no weights, no network, no Elasticsearch."""
    return settings_factory(
        embedder_backend="fake",
        reranker_backend="fake",
        llm_backend="fake",
        vector_store_backend="fake",
    )


@pytest.fixture
async def fake_deps(fake_settings: RagSettings) -> AsyncIterator[RagDependencies]:
    """The four fake ports, opened from settings over the seeded in-memory store."""
    async with RagPorts(fake_settings) as ports:
        yield await ports.workflow_dependencies()
