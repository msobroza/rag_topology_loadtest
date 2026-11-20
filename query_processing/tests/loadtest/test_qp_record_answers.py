"""``qp-record-answers``: what it sends, what it keeps, and when it writes nothing.

It writes the answer record's new form whichever shape its target answered
in: the vector-store records under ``passages`` or under ``records``.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import httpx
import pytest

from answer_record_helpers import (
    NEW_BODY,
    OLD_BODY,
    SCORES,
    old_entry,
    record,
    use_loadtest_variables,
    workflow_client,
)
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.retrieval.corpus import synthetic_questions
from query_processing.loadtest.answer_record import read_answer_record
from query_processing.loadtest.qp_record_answers import record_answers_main
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
HOST = "http://target.test"


def recorded_with(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings_factory: SettingsFactory,
    variables: dict[str, str],
) -> Path:
    """Record the fake app under ``variables``; return the record's prefix."""
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "run") == 0
    return tmp_path / "run"


def test_the_record_holds_no_timing_no_tokens_no_request_id_and_no_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings_factory: SettingsFactory
) -> None:
    prefix = recorded_with(monkeypatch, tmp_path, settings_factory, {})

    text = Path(f"{prefix}_answers.json").read_text(encoding="utf-8")
    answer = json.loads(text)["answers"][1]

    assert answer["status"] == 200
    assert answer["records"]
    for left_out in ("timings_ms", "tokens", "request_id", "run_label", "question"):
        assert left_out not in answer
    assert set(answer["records"][0]) == {
        "text_sha256", "metadata", "retrieval_score", "rerank_score"
    }  # fmt: skip
    assert isinstance(answer["records"][0]["metadata"]["id"], str)
    assert read_answer_record(prefix).seen[0].responses == 6


@pytest.mark.parametrize(
    ("variables", "endpoints"),
    [
        ({}, ["/retrieve", "/query"]),
        ({"QP_LOADTEST_QUERY_WEIGHT": "0"}, ["/retrieve"]),
        ({"QP_LOADTEST_RETRIEVE_WEIGHT": "0"}, ["/query"]),
    ],
    ids=["both", "retrieve-alone", "query-alone"],
)
def test_each_body_goes_to_each_endpoint_of_a_weight_above_zero_in_the_pools_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings_factory: SettingsFactory,
    variables: dict[str, str],
    endpoints: list[str],
) -> None:
    prefix = recorded_with(monkeypatch, tmp_path, settings_factory, variables)

    answers = read_answer_record(prefix).answers

    pool = list(synthetic_questions(3))
    assert [(a.endpoint, a.body) for a in answers] == [
        (endpoint, {"question": question})
        for question in pool
        for endpoint in endpoints
    ]
    assert all(a.status == 200 for a in answers)


def test_both_weights_at_zero_are_refused_naming_both_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_loadtest_variables(
        monkeypatch,
        tmp_path,
        {"QP_LOADTEST_QUERY_WEIGHT": "0", "QP_LOADTEST_RETRIEVE_WEIGHT": "0"},
    )
    never = httpx.Client(transport=httpx.MockTransport(pytest.fail))

    status = record_answers_main(["--host", HOST, "--out", "run"], client=never)

    error = capsys.readouterr().err
    assert status == 2
    assert "QP_LOADTEST_RETRIEVE_WEIGHT=0" in error
    assert "QP_LOADTEST_QUERY_WEIGHT=0" in error
    assert not list(tmp_path.glob("*_answers.json"))


def test_a_target_that_never_answers_is_retried_as_the_settings_say_then_nothing_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_loadtest_variables(
        monkeypatch,
        tmp_path,
        {"QP_LOADTEST_RETRIES": "2", "QP_LOADTEST_BACKOFF_S": "0.5"},
    )
    attempts: list[str] = []
    waits: list[float] = []

    def unreachable(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(unreachable))
    argv = ["--host", HOST, "--out", str(tmp_path / "run")]

    status = record_answers_main(argv, client=client, sleep=waits.append)

    assert status == 3
    assert attempts == [f"{HOST}/retrieve"] * 3  # the first request, then it stops
    assert waits == [0.5, 1.0]  # the backoff doubles
    assert "nothing written" in capsys.readouterr().err
    assert not list(tmp_path.glob("*_answers.json"))


def test_a_response_that_is_no_json_object_is_recorded_with_its_status_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": "1"})
    answers = iter([httpx.Response(200, json=[1]), httpx.Response(502, text="busy")])
    client = httpx.Client(transport=httpx.MockTransport(lambda _: next(answers)))

    status = record_answers_main(["--host", HOST, "--out", "run"], client=client)

    recorded = read_answer_record(tmp_path / "run").answers
    assert status == 0
    assert [(a.status, a.mode, a.records) for a in recorded] == [
        (200, None, None),
        (502, None, None),
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "retrieve", "passages": [{"text": "no id"}]},
        {"mode": "retrieve", "records": [{"text": "no id", "metadata": {}}]},
    ],
    ids=["passages", "records"],
)
def test_a_200_not_of_the_answers_shape_stops_the_command_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: dict[str, Any],
) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": "1"})

    status = record_answers_main(
        ["--host", HOST, "--out", "run"], client=answering(body)
    )

    error = capsys.readouterr().err
    assert status == 4
    assert "/retrieve: records.0.metadata: " in error
    assert '"records"?: [' in error  # the shape expected, in both its forms
    assert '"passages"?: [' in error
    assert "nothing written" in error
    assert not list(tmp_path.glob("*_answers.json"))


def test_the_load_tests_credentials_are_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_loadtest_variables(
        monkeypatch,
        tmp_path,
        {"QP_LOADTEST_QUESTIONS": "1", "QP_LOADTEST_BEARER_TOKEN": "run-token"},
    )
    sent: list[str | None] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(answer))

    assert record_answers_main(["--host", HOST, "--out", "run"], client=client) == 0
    assert sent == ["Bearer run-token"] * 2


def test_without_a_questions_file_it_reads_no_file_and_calls_its_target_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The check is self-contained: the synthetic questions are built in
    # process, and the only call out is the command's own client's.
    use_loadtest_variables(monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": "2"})
    called: list[str] = []
    read: list[str] = []
    real_open = io.open

    def answer(_: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        called.append(str(request.url))
        return httpx.Response(200, json={})

    def watched_open(file: Any, mode: str = "r", *args: Any, **kw: Any) -> IO[Any]:
        if "r" in mode:
            read.append(str(file))
        return real_open(file, mode, *args, **kw)

    def no_connection(*_: object) -> None:
        raise AssertionError("a connection was opened")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", answer)
    monkeypatch.setattr(socket.socket, "connect", no_connection)
    monkeypatch.setattr(io, "open", watched_open)
    monkeypatch.setattr(builtins, "open", watched_open)

    status = record_answers_main(["--host", HOST, "--out", "run"])

    assert status == 0
    assert read == []
    assert called == [f"{HOST}/retrieve", f"{HOST}/query"] * 2


def answering(body: dict[str, Any]) -> httpx.Client:
    """A client of a target that answers ``body`` to every request."""
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )


def recorded_answer_of(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict[str, Any]
) -> dict[str, Any]:
    """The JSON of the one answer recorded of a target answering ``body``."""
    use_loadtest_variables(
        monkeypatch,
        tmp_path,
        {"QP_LOADTEST_QUESTIONS": "1", "QP_LOADTEST_QUERY_WEIGHT": "0"},
    )
    argv = ["--host", HOST, "--out", "run"]
    assert record_answers_main(argv, client=answering(body)) == 0
    text = Path(f"{tmp_path / 'run'}_answers.json").read_text(encoding="utf-8")
    return json.loads(text)["answers"][0]


# The new shape from an index ingested before: its metadata keeps ``chunk``.
OLD_INDEX_BODY = {
    **NEW_BODY,
    "records": [
        {
            **entry,
            "metadata": {"doc_id": "doc-1", "chunk": index, "id": f"doc-1#{index}"},
        }
        for index, entry in enumerate(NEW_BODY["records"])
    ],
}


@pytest.mark.parametrize(
    "body",
    [OLD_BODY, NEW_BODY, OLD_INDEX_BODY],
    ids=["passages", "records", "records-of-an-index-ingested-before"],
)
def test_a_response_of_either_shape_is_written_in_the_new_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict[str, Any]
) -> None:
    answer = recorded_answer_of(monkeypatch, tmp_path, body)

    assert "passages" not in answer
    assert answer["records"] == [
        {
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "metadata": {"doc_id": "doc-1", "sequence": index, "id": f"doc-1#{index}"},
            **SCORES,
        }
        for index, text in enumerate(["first", "second"])
    ]


@pytest.mark.parametrize(
    "chunk",
    ["a chunk's own payload", True],
    ids=["a-text-is-a-chunk", "a-boolean-is-no-whole-number"],
)
def test_only_a_whole_number_under_chunk_is_read_as_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunk: object
) -> None:
    body = {"mode": "retrieve", "passages": [old_entry("doc-1#0", "first", chunk)]}

    answer = recorded_answer_of(monkeypatch, tmp_path, body)

    assert answer["records"][0]["metadata"] == {
        "doc_id": "doc-1", "chunk": chunk, "id": "doc-1#0"
    }  # fmt: skip
