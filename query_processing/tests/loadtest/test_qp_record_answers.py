"""``qp-record-answers``: what it sends, what it keeps, and when it writes nothing."""

from __future__ import annotations

import builtins
import io
import json
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import httpx
import pytest

from answer_record_helpers import record, use_loadtest_variables, workflow_client
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.retrieval.corpus import synthetic_questions
from query_processing.loadtest.answer_record import read_answer_record
from query_processing.loadtest.qp_record_answers import record_answers_main
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
HOST = "http://target.test"
TOKEN_URL = "http://domino.test/access-token"


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
    assert answer["passages"]
    for left_out in ("timings_ms", "tokens", "request_id", "run_label", "question"):
        assert left_out not in answer
    assert set(answer["passages"][0]) == {
        "id", "text_sha256", "metadata", "retrieval_score", "rerank_score"
    }  # fmt: skip
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
    assert [(a.status, a.mode, a.passages) for a in recorded] == [
        (200, None, None),
        (502, None, None),
    ]


def test_a_200_not_of_the_answers_shape_stops_the_command_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": "1"})
    body = {"mode": "retrieve", "passages": [{"text": "no id"}]}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )

    status = record_answers_main(["--host", HOST, "--out", "run"], client=client)

    error = capsys.readouterr().err
    assert status == 4
    assert "/retrieve: passages.0.id: " in error
    assert "nothing written" in error
    assert not list(tmp_path.glob("*_answers.json"))


ENV_KEY = {"QP_LOADTEST_KEY__SOURCE": "env", "QP_LOADTEST_KEY__VARIABLE": "TARGET_KEY"}
DOMINO_KEY = {"QP_LOADTEST_KEY__SOURCE": "domino", "QP_DOMINO__TOKEN_URL": TOKEN_URL}


@pytest.mark.parametrize(
    ("variables", "header", "value"),
    [
        ({**ENV_KEY, "TARGET_KEY": "k-1"}, "Authorization", "Bearer k-1"),
        ({"DOMINO_API_KEY": "dk"}, "X-Domino-Api-Key", "dk"),
        (
            {**ENV_KEY, "TARGET_KEY": "k-1", "DOMINO_API_KEY": "dk"},
            "Authorization",
            "Bearer k-1",
        ),
        ({}, "Authorization", None),
    ],
    ids=["env", "domino-api-key", "a-key-block-wins", "none"],
)
def test_each_request_carries_the_key_the_settings_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variables: dict[str, str],
    header: str,
    value: str | None,
) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": "1"})
    for name, given in variables.items():
        monkeypatch.setenv(name, given)
    target = KeyedTarget()

    assert target.record() == 0
    assert target.carried(header) == [value] * 2
    if "DOMINO_API_KEY" in variables and value != "dk":
        assert target.carried("X-Domino-Api-Key") == [None] * 2


def test_the_domino_source_sends_the_run_token_held_for_its_lifetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_loadtest_variables(
        monkeypatch, tmp_path, {**DOMINO_KEY, "QP_LOADTEST_QUESTIONS": "3"}
    )
    target = KeyedTarget()

    assert target.record() == 0
    assert target.tokens_served == 1  # within QP_DOMINO__TOKEN_TTL_S's 240 s
    assert target.carried("Authorization") == ["Bearer run-token-1"] * 6


def test_the_run_token_is_asked_for_again_once_its_lifetime_is_over(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variables = {**DOMINO_KEY, "QP_DOMINO__TOKEN_TTL_S": "0.001"}
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    # Each answer takes longer than a token is held.
    target = KeyedTarget(answer_s=0.002)

    assert target.record() == 0
    assert target.tokens_served == 6
    assert target.carried("Authorization") == [
        f"Bearer run-token-{n}" for n in range(1, 7)
    ]


def test_a_401_tells_the_source_and_is_sent_again_with_the_new_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_loadtest_variables(
        monkeypatch,
        tmp_path,
        {**DOMINO_KEY, "QP_LOADTEST_QUESTIONS": "1", "QP_LOADTEST_BACKOFF_S": "0.5"},
    )
    target = KeyedTarget(refused={"Bearer run-token-1"})

    assert target.record() == 0
    assert target.carried("Authorization") == [
        "Bearer run-token-1",  # refused: the source is told
        "Bearer run-token-2",
        "Bearer run-token-2",
    ]
    assert target.waits == [0.5]  # within the retries, after their backoff
    assert [a.status for a in read_answer_record(tmp_path / "run").answers] == [200] * 2


def test_when_the_retries_run_out_on_401s_the_last_response_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variables = {**ENV_KEY, "TARGET_KEY": "rotated", "QP_LOADTEST_QUESTIONS": "1"}
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    target = KeyedTarget(refused={"Bearer rotated"})

    assert target.record() == 0
    assert len(target.carried("Authorization")) == 2 * 3  # QP_LOADTEST_RETRIES=2
    assert [a.status for a in read_answer_record(tmp_path / "run").answers] == [401] * 2


def test_a_401_without_a_key_source_is_recorded_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variables = {"DOMINO_API_KEY": "dk", "QP_LOADTEST_QUESTIONS": "1"}
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    target = KeyedTarget(refused={None})

    assert target.record() == 0
    assert target.carried("X-Domino-Api-Key") == ["dk"] * 2
    assert target.waits == []


@pytest.mark.parametrize(
    ("variables", "line"),
    [
        (
            {"QP_LOADTEST_KEY__SOURCE": "vault"},
            "QP_LOADTEST_KEY__SOURCE: unknown key source 'vault'",
        ),
        (
            {"QP_LOADTEST_KEY__SOURCE": "env"},
            "QP_LOADTEST_KEY: the 'env' key source reads 'variable' beside "
            "'source'; got none",
        ),
        (
            {**DOMINO_KEY, "QP_LOADTEST_KEY__VARIABLE": "X"},
            "QP_LOADTEST_KEY: the 'domino' key source reads no key beside "
            "'source'; got 'variable'",
        ),
        (ENV_KEY, "the environment variable TARGET_KEY is unset or empty"),
    ],
    ids=["unknown-source", "missing-key", "unread-key", "unset-variable"],
)
def test_a_key_block_start_up_would_refuse_is_refused_before_a_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    variables: dict[str, str],
    line: str,
) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    never = httpx.MockTransport(pytest.fail)

    argv = ["--host", HOST, "--out", "run"]
    status = record_answers_main(
        argv, client=httpx.Client(transport=never), key_transport=never
    )

    [error] = capsys.readouterr().err.splitlines()
    assert status == 2
    assert error.startswith(f"qp-record-answers: {line}")
    assert not list(tmp_path.glob("*_answers.json"))


def test_a_token_url_that_never_answers_is_asked_within_the_load_tests_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    variables = {**DOMINO_KEY, "QP_LOADTEST_RETRIES": "1", "QP_LOADTEST_BACKOFF_S": "0"}
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    asked: list[str] = []

    def down(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return httpx.Response(503, text="token service down")

    transport = httpx.MockTransport(down)
    argv = ["--host", HOST, "--out", "run"]
    status = record_answers_main(
        argv, client=httpx.Client(transport=transport), key_transport=transport
    )

    assert status == 3
    assert asked == [TOKEN_URL] * 2  # QP_LOADTEST_RETRIES=1: two asks
    assert TOKEN_URL in capsys.readouterr().err
    assert not list(tmp_path.glob("*_answers.json"))


class KeyedTarget:
    """A target and a Domino token URL in one stand-in transport, no network.

    The token URL serves ``run-token-1``, then ``run-token-2``, ...; the target
    answers 401 to a request whose ``Authorization`` is in ``refused`` and 200
    to any other, taking at least ``answer_s`` to answer.
    """

    def __init__(
        self, refused: set[str | None] | None = None, answer_s: float = 0.0
    ) -> None:
        self.tokens_served, self.waits = 0, []
        self.requests: list[httpx.Request] = []
        self._refused, self._answer_s = refused or set(), answer_s
        self.transport = httpx.MockTransport(self._answer)

    def record(self) -> int:
        """Run ``qp-record-answers`` against this target; return its status."""
        argv = ["--host", HOST, "--out", "run"]
        client = httpx.Client(transport=self.transport)
        return record_answers_main(
            argv, client=client, sleep=self.waits.append, key_transport=self.transport
        )

    def carried(self, header: str) -> list[str | None]:
        """What each request to the target carried in ``header``, in order."""
        return [request.headers.get(header) for request in self.requests]

    def _answer(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.tokens_served += 1
            return httpx.Response(200, text=f"run-token-{self.tokens_served}")
        self.requests.append(request)
        time.sleep(self._answer_s)
        if request.headers.get("Authorization") in self._refused:
            return httpx.Response(401, text="expired")
        return httpx.Response(200, json={})


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
