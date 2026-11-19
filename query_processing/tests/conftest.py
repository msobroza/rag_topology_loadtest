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

from query_processing.application.rag_dependencies import RagDependencies
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.files.prompts_file import PROMPTS_FILE_VAR
from query_processing.config.files.workflow_file import (
    WORKFLOW_FILE_VARIABLE,
    unknown_keys,
)
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.testing import add_pin_answers_option, pin_answers_asked
from rag_components import FAKE_BACKENDS, settings_at_paths
from rag_graphs import FIXTURE_PROMPTS, FIXTURE_WORKFLOW

# The toolkit ships no workflow file, and no test of it reads a use case's: this
# is the one place the tests are given theirs (helpers/fixture_workflow.yaml,
# made-up models, and its helpers/fixture_prompts.yaml). Set here for what a
# test module computes as it is collected, and again around every test below.
os.environ[WORKFLOW_FILE_VARIABLE] = str(FIXTURE_WORKFLOW)
os.environ[PROMPTS_FILE_VAR] = str(FIXTURE_PROMPTS)


def pytest_addoption(parser: pytest.Parser) -> None:
    """``--pin-answers``: ``make pin-answers`` makes the pinned answer record again."""
    add_pin_answers_option(parser)


@pytest.fixture
def pin_answers(request: pytest.FixtureRequest) -> bool:
    """Whether the run makes each pinned answer record again (``--pin-answers``)."""
    return pin_answers_asked(request.config)


@pytest.fixture(autouse=True)
def _the_fixture_workflow_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test reads the tests' own workflow and prompts files, unless it says otherwise.

    A ``workflow.yaml`` left in the working directory, or ``QP_WORKFLOW_FILE``
    set in the shell or by an earlier test, would change the graph the tests
    run. A test names another file with the variable, or unsets it to find one
    in the working directory; an autouse fixture is set up before the test's own.
    The prompts file likewise, by ``QP_PROMPTS_FILE``.
    """
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(FIXTURE_WORKFLOW))
    monkeypatch.setenv(PROMPTS_FILE_VAR, str(FIXTURE_PROMPTS))


@pytest.fixture
def settings_factory() -> Callable[..., WorkflowSettings]:
    """WorkflowSettings without reading a stray ``.env``; keyword overrides allowed.

    A keyword names a setting, or a value of the workflow file by its path
    (``components__reranker__backend``); either is merged into the file. A
    top-level key that is no setting is refused: the settings would ignore it
    silently, and a test that meant a backend set to fake under a name since
    removed would run on the file's real backends.
    """

    def make(**overrides: object) -> WorkflowSettings:
        values = settings_at_paths(**overrides)
        unknown = unknown_keys(values, WorkflowSettings)
        if unknown:
            raise TypeError(
                f"settings_factory: {', '.join(map(repr, unknown))} is no setting; "
                "expected a field of WorkflowSettings or a path below one, such as "
                "components__vector_store__backend"
            )
        return WorkflowSettings(_env_file=None, **values)

    return make


@pytest.fixture
def fake_settings(
    settings_factory: Callable[..., WorkflowSettings],
) -> WorkflowSettings:
    """Every backend set to ``fake``: no weights, no network, no Elasticsearch."""
    return settings_factory(**FAKE_BACKENDS)


@pytest.fixture
async def fake_deps(fake_settings: WorkflowSettings) -> AsyncIterator[RagDependencies]:
    """The four fake ports, opened from settings over the seeded in-memory store."""
    async with CompositionRoot(fake_settings) as composition:
        yield await composition.workflow_dependencies()
