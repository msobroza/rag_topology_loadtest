"""Shared fixtures of the RAG use case's tests: its own files, and no model weights.

The tests import only what the toolkit installs (``rag_load_test``,
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

USE_CASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def use_case_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from the use case's directory, with no RAG_* or DOMINO_* of the shell.

    A variable left in the shell, such as ``RAG_WORKFLOW_FILE``, would point a
    command away from the use case's own files.
    """
    for name in [n for n in os.environ if n.startswith(("RAG_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(USE_CASE_DIR)
    return USE_CASE_DIR
