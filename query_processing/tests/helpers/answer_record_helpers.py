"""Record a workflow app's answers in process, and compare the records.

The app is the workflow app of ``qp-serve`` on settings a test gives, started
with ``TestClient``, which is an ``httpx.Client``: ``qp-record-answers`` is
handed it, so no socket is opened. The load test's variables are cleared and
no ``.env`` is read, so a test names every variable it sets.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from loadtest_settings_helpers import clear_loadtest_variables
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app
from query_processing.loadtest.qp_compare_answers import compare_answers_main
from query_processing.loadtest.qp_record_answers import record_answers_main

QUESTIONS = "3"  # synthetic questions a record holds, each sent to both endpoints


def use_loadtest_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variables: Mapping[str, str]
) -> None:
    """Run in ``tmp_path`` (no ``.env``) with exactly these load-test variables.

    ``QP_LOADTEST_QUESTIONS`` is ``QUESTIONS`` unless ``variables`` names it.
    """
    clear_loadtest_variables(monkeypatch)
    monkeypatch.chdir(tmp_path)
    for name, value in {"QP_LOADTEST_QUESTIONS": QUESTIONS, **variables}.items():
        monkeypatch.setenv(name, value)


@contextmanager
def workflow_client(
    settings: WorkflowSettings, transport: httpx.AsyncBaseTransport | None = None
) -> Iterator[TestClient]:
    """The workflow app on ``settings``, started; its calls out go to ``transport``."""
    composition = CompositionRoot(settings, transport=transport)
    with TestClient(create_app(composition)) as client:
        yield client


def record(client: httpx.Client, prefix: Path) -> int:
    """Run ``qp-record-answers`` against ``client``'s app; return its status."""
    argv = ["--host", str(client.base_url), "--out", str(prefix)]
    return record_answers_main(argv, client=client)


def compare(*prefixes: Path, out: Path) -> int:
    """Run ``qp-compare-answers`` over ``prefixes`` into ``out``; return its status."""
    return compare_answers_main([*map(str, prefixes), "--out", str(out)])
