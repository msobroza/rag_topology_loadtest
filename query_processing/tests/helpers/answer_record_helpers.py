"""Record a workflow app's answers in process, and compare the records.

The code is ``query_processing.testing``'s (``pinned_answer_record``), which a
use case's tests share; this module gives the toolkit's tests its names and
their default count of questions. The app is the workflow app of
``qp-serve`` on settings a test gives, started with ``TestClient``, which is
an ``httpx.Client``: ``qp-record-answers`` is handed it, so no socket is
opened. The load test's variables are cleared and no ``.env`` is read, so a
test names every variable it sets.

The workflow app answered with its vector-store records under ``passages``,
each with its ``id`` beside its metadata and the handbook's window index under
the metadata key ``chunk``; it answers with them under ``records``, the id in
the metadata and the index under ``sequence``. ``OLD_BODY`` and ``NEW_BODY``
are one answer in each shape, and ``old_form_of`` writes an answer record of
the new form in the old one, as the registry's are.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from query_processing.testing import compare_answer_records as compare
from query_processing.testing import record_answers_of as record
from query_processing.testing import use_loadtest_variables as use_variables
from query_processing.testing import workflow_client

QUESTIONS = "3"  # synthetic questions a record holds, each sent to both endpoints
SCORES = {"retrieval_score": 0.75, "rerank_score": 0.5}

__all__ = [
    "NEW_BODY",
    "OLD_BODY",
    "QUESTIONS",
    "SCORES",
    "compare",
    "old_entry",
    "old_form_of",
    "record",
    "use_loadtest_variables",
    "workflow_client",
]


def old_entry(record_id: str, text: str, chunk: object) -> dict[str, Any]:
    """A vector-store record as the workflow app answered it before, under ``passages``."""
    metadata = {"doc_id": "doc-1", "chunk": chunk}
    return {"id": record_id, "text": text, "metadata": metadata, **SCORES}


def new_entry(record_id: str, text: str, sequence: int) -> dict[str, Any]:
    """The same vector-store record as the workflow app answers it now, under ``records``."""
    metadata = {"doc_id": "doc-1", "sequence": sequence, "id": record_id}
    return {"text": text, "metadata": metadata, **SCORES}


OLD_BODY: dict[str, Any] = {
    "mode": "retrieve",
    "metadata_filter": [],
    "answer": None,
    "passages": [old_entry("doc-1#0", "first", 0), old_entry("doc-1#1", "second", 1)],
}
NEW_BODY: dict[str, Any] = {
    "mode": "retrieve",
    "metadata_filter": [],
    "answer": None,
    "records": [new_entry("doc-1#0", "first", 0), new_entry("doc-1#1", "second", 1)],
}


def old_form_of(document: dict[str, Any]) -> dict[str, Any]:
    """The JSON of an answer record of the new form, written in the old form."""
    for answer in document["answers"]:
        entries = answer.pop("records")
        answer["passages"] = None if entries is None else list(map(_old, entries))
    return document


def _old(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(entry["metadata"])
    record_id = metadata.pop("id")
    if "sequence" in metadata:
        metadata["chunk"] = metadata.pop("sequence")
    return {"id": record_id, **entry, "metadata": metadata}


def use_loadtest_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variables: Mapping[str, str]
) -> None:
    """Run in ``tmp_path`` (no ``.env``) with exactly these load-test variables.

    ``QP_LOADTEST_QUESTIONS`` is ``QUESTIONS`` unless ``variables`` names it.
    """
    use_variables(
        monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": QUESTIONS, **variables}
    )
