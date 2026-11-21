"""``qp-record-answers`` then ``qp-compare-answers``: what answers the same compares so.

Each test records the workflow app in process, on the fixture workflow file
with every component on ``fake``, and compares the records. A record of the
old form, its vector-store records under ``passages`` as the registry's are,
is read through the same mapping as a response of the old shape, and compares
alike with a record of the new form of the same answers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from answer_record_helpers import (
    compare,
    old_form_of,
    record,
    use_loadtest_variables,
    workflow_client,
)
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_model_server import create_model_server
from query_processing.loadtest.answer_record import read_answer_record
from query_processing.loadtest.qp_record_answers import record_answers_main
from rag_components import FAKE_BACKENDS
from repo_paths import REPO_ROOT

SettingsFactory = Callable[..., WorkflowSettings]
RERANK_URL = "http://reranker-app.test/v3/rerank"
REGISTRY_RECORD = (
    REPO_ROOT
    / "experiments/2026-11-17-answers-of-main-and-the-toolkit-on-real-models"
    / "records/main-monolith"
)  # the --out prefix of one answer record of 17 November


def rewritten(prefix: Path, name: str, edit: Callable[[Any], Any]) -> Path:
    """A record named ``name``: the one under ``prefix`` with ``edit`` applied."""
    document = json.loads(Path(f"{prefix}_answers.json").read_text(encoding="utf-8"))
    other = prefix.with_name(name)
    Path(f"{other}_answers.json").write_text(
        json.dumps(edit(document)), encoding="utf-8"
    )
    return other


def test_a_record_of_the_old_form_compares_alike_with_one_of_the_new(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "new") == 0
    old = rewritten(tmp_path / "new", "old", old_form_of)

    status = compare(old, tmp_path / "new", out=tmp_path / "c.md")

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert '"passages": [' in Path(f"{old}_answers.json").read_text(encoding="utf-8")
    assert status == 0, report
    assert "| new | 6 | 6 | 0 |" in report


def test_a_difference_from_a_record_of_the_old_form_is_named_in_the_new_forms_words(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    def move_a_score(document: Any) -> Any:
        document["answers"][1]["records"][0]["rerank_score"] += 0.5
        return document

    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "new") == 0
    old = rewritten(tmp_path / "new", "old", old_form_of)
    run = rewritten(tmp_path / "new", "run", move_a_score)

    status = compare(old, run, out=tmp_path / "c.md")

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 1
    assert "| records[0].rerank_score |" in report
    assert "| passages[" not in report


def test_an_answer_record_of_17_november_is_read_through_the_mapping_unchanged() -> (
    None
):
    path = Path(f"{REGISTRY_RECORD}_answers.json")
    before = path.read_bytes()

    answers = read_answer_record(REGISTRY_RECORD).answers

    entries = [entry for answer in answers for entry in answer.records or []]
    assert (len(answers), len(entries)) == (50, 500)
    assert all(isinstance(entry.metadata["id"], str) for entry in entries)
    assert all(isinstance(entry.metadata["sequence"], int) for entry in entries)
    assert not any("chunk" in entry.metadata for entry in entries)
    assert path.read_bytes() == before


@pytest.fixture(autouse=True)
def _loadtest_variables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, {})


def test_the_same_app_recorded_twice_compares_as_identical(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "before") == 0
        assert record(client, tmp_path / "after") == 0

    status = compare(tmp_path / "before", tmp_path / "after", out=tmp_path / "c.md")

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 0
    assert "| after | 6 | 6 | 0 |" in report


def test_the_app_under_two_topology_aliases_compares_as_identical(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    for alias in ("monolith", "split-all"):
        settings = settings_factory(**FAKE_BACKENDS, topology_alias=alias)
        with workflow_client(settings) as client:
            assert record(client, tmp_path / alias) == 0

    status = compare(
        tmp_path / "monolith", tmp_path / "split-all", out=tmp_path / "c.md"
    )

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 0
    assert "topology `split-all`" in report
    assert "| split-all | 6 | 6 | 0 |" in report


def test_a_reranker_served_by_the_model_server_answers_as_the_one_in_process(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "in-process") == 0
    model_server = create_model_server(
        "reranker",
        CompositionRoot(settings_factory(**FAKE_BACKENDS), reranker=FakeReranker()),
    )
    served = settings_factory(
        **{
            **FAKE_BACKENDS,
            "components__reranker__backend": "served",
            "components__reranker__served__url": RERANK_URL,
        }
    )
    with TestClient(model_server):  # runs its lifespan
        transport = httpx.ASGITransport(app=model_server)
        with workflow_client(served, transport) as client:
            assert record(client, tmp_path / "served") == 0

    status = compare(
        tmp_path / "in-process", tmp_path / "served", out=tmp_path / "c.md"
    )

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 0, report
    assert "reranker=served" in report  # the run label: the call went over HTTP
    assert "| served | 6 | 6 | 0 |" in report


def a_chunk_answered(chunk_text: str) -> httpx.Client:
    """A target answering one vector-store record made from a chunk of ``chunk_text``."""
    metadata = {
        "id": "doc-handbook#0",
        "chunk": json.dumps({"id": "doc-handbook#0", "text": chunk_text}),
        "chunk__context__text": f"The context of: {chunk_text}",
        "sequence": 0,
    }
    entry = {"text": chunk_text, "metadata": metadata, "retrieval_score": 0.5}
    body = {"mode": "retrieve", "records": [{**entry, "rerank_score": None}]}
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )


def recorded_from(target: httpx.Client, prefix: Path) -> Path:
    """Record ``target`` under ``prefix``; return the prefix."""
    argv = ["--host", "http://target.test", "--out", str(prefix)]
    assert record_answers_main(argv, client=target) == 0
    return prefix


@pytest.mark.parametrize(
    ("other_text", "expected"),
    [("Twenty paid vacation days.", 0), ("Twenty-five paid vacation days.", 1)],
    ids=["the-same-chunk-compares-alike", "another-chunk-differs"],
)
def test_two_records_of_chunks_compare_by_the_chunks_hashes(
    tmp_path: Path, other_text: str, expected: int
) -> None:
    before = recorded_from(
        a_chunk_answered("Twenty paid vacation days."), tmp_path / "a"
    )
    after = recorded_from(a_chunk_answered(other_text), tmp_path / "b")

    status = compare(before, after, out=tmp_path / "c.md")

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == expected, report
    assert ("| records[0].metadata |" in report) is bool(expected)


def written_before(document: Any) -> Any:
    """The record as it was written before: each chunk's texts whole, not hashed."""
    for answer in document["answers"]:
        metadata = answer["records"][0]["metadata"]
        del metadata["chunk_sha256"], metadata["chunk__context__text_sha256"]
        metadata["chunk"] = json.dumps({"id": "doc-handbook#0", "text": "Twenty."})
        metadata["chunk__context__text"] = "The context of: Twenty."
    return document


def test_a_record_written_before_holding_a_chunks_texts_compares_alike_with_one_now(
    tmp_path: Path,
) -> None:
    # A record of the old form held no chunk (the 17 November test above).
    now = recorded_from(a_chunk_answered("Twenty."), tmp_path / "now")
    before = rewritten(now, "before", written_before)

    status = compare(before, now, out=tmp_path / "c.md")

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "Twenty." in Path(f"{before}_answers.json").read_text(encoding="utf-8")
    assert status == 0, report
