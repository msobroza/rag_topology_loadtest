"""Shared fixtures of the RAG use case's tests: its own files, and no model weights.

The tests import only what the toolkit installs (``query_processing``,
``fake_domino``), never the toolkit's tests, and run each command from this use
case's directory, as its Makefile does.
"""

from __future__ import annotations

import os
import tempfile

# No test may reach real model weights: the Hub is offline and its cache an
# empty folder, so a component left on its local backend fails at once instead
# of downloading gigabytes (as the toolkit's suite guards itself).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "rag-tests-empty-hf-home")
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

from pathlib import Path

import pytest
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.testing import add_pin_answers_option, pin_answers_asked

from workflow_on_fakes import all_fake_settings

USE_CASE_DIR = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    """``--pin-answers``: ``make pin-answers`` makes the pinned answer records again."""
    add_pin_answers_option(parser)


@pytest.fixture
def pin_answers(request: pytest.FixtureRequest) -> bool:
    """Whether the run makes each pinned answer record again (``--pin-answers``)."""
    return pin_answers_asked(request.config)


@pytest.fixture
def use_case_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from the use case's directory, with no QP_* or DOMINO_* of the shell.

    A variable left in the shell, such as ``QP_WORKFLOW_FILE``, would point a
    command away from the use case's own files.
    """
    for name in [n for n in os.environ if n.startswith(("QP_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(USE_CASE_DIR)
    return USE_CASE_DIR


@pytest.fixture
def all_fake(use_case_dir: Path, monkeypatch: pytest.MonkeyPatch) -> WorkflowSettings:
    """This directory's workflow file, every component on its fake backend."""
    del use_case_dir  # requested so the settings are read from this directory
    return all_fake_settings(monkeypatch)
