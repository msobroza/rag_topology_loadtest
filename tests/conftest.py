"""Shared fixtures. No model weights, no network, no Elasticsearch anywhere in this suite."""

from __future__ import annotations

import os
import tempfile

# locust gevent-patches the interpreter at import unless this is set; it must run
# before any ``import locust`` (the locustfile tests import it).
os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

# No test may reach real model weights. The Hub is offline and its cache is an
# empty folder, so a test that opens an in-process model without a fake fails at
# once. Without this it downloads the model, or loads a cached one and exports
# it: gigabytes, twice on 2026-11-15, when tests passed settings a change had
# removed, the settings ignored them, and the app ran on its `local` backends.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "rag-tests-empty-hf-home")
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

from collections.abc import AsyncIterator, Callable

import pytest

from rag_components import FAKE_BACKENDS, settings_at_paths
from rag_load_test.application.rag_dependencies import RagDependencies
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import (
    PACKAGED_WORKFLOW,
    WORKFLOW_FILE_VARIABLE,
)


@pytest.fixture(autouse=True)
def _the_packaged_workflow_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test reads the workflow file that ships with the code, unless it says.

    A ``workflow.yaml`` left in the working directory, or ``RAG_WORKFLOW_FILE``
    set in the shell, would change the graph the tests run. A test names
    another file with the variable, or unsets it to find one in the working
    directory; an autouse fixture is set up before the test's own.
    """
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(PACKAGED_WORKFLOW))


@pytest.fixture
def settings_factory() -> Callable[..., RagSettings]:
    """RagSettings without reading a stray ``.env``; keyword overrides allowed.

    A keyword names a setting, or a value of the workflow file by its path
    (``components__reranker__backend``); either is merged into the file.
    """

    def make(**overrides: object) -> RagSettings:
        return RagSettings(_env_file=None, **settings_at_paths(**overrides))

    return make


@pytest.fixture
def fake_settings(settings_factory: Callable[..., RagSettings]) -> RagSettings:
    """Every backend set to ``fake``: no weights, no network, no Elasticsearch."""
    return settings_factory(**FAKE_BACKENDS)


@pytest.fixture
async def fake_deps(fake_settings: RagSettings) -> AsyncIterator[RagDependencies]:
    """The four fake ports, opened from settings over the seeded in-memory store."""
    async with CompositionRoot(fake_settings) as composition:
        yield await composition.workflow_dependencies()
