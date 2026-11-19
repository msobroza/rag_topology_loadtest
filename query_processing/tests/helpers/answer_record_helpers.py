"""Record a workflow app's answers in process, and compare the records.

The code is ``query_processing.testing``'s (``pinned_answer_record``), which a
use case's tests share; this module gives the toolkit's tests its names and
their default count of questions. The app is the workflow app of
``qp-serve`` on settings a test gives, started with ``TestClient``, which is
an ``httpx.Client``: ``qp-record-answers`` is handed it, so no socket is
opened. The load test's variables are cleared and no ``.env`` is read, so a
test names every variable it sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from query_processing.testing import compare_answer_records as compare
from query_processing.testing import record_answers_of as record
from query_processing.testing import use_loadtest_variables as use_variables
from query_processing.testing import workflow_client

QUESTIONS = "3"  # synthetic questions a record holds, each sent to both endpoints

__all__ = [
    "QUESTIONS",
    "compare",
    "record",
    "use_loadtest_variables",
    "workflow_client",
]


def use_loadtest_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variables: Mapping[str, str]
) -> None:
    """Run in ``tmp_path`` (no ``.env``) with exactly these load-test variables.

    ``QP_LOADTEST_QUESTIONS`` is ``QUESTIONS`` unless ``variables`` names it.
    """
    use_variables(
        monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": QUESTIONS, **variables}
    )
