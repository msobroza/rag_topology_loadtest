"""``qp-record-answers`` and the key its requests carry, from a key source.

Each request carries the key the load test's settings name, a 401 tells the
source and is sent again within the load test's retries, and a key block
start-up would refuse is a usage error that prints no key. Out of
test_qp_record_answers.py for the 500-line limit of a file.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
from test_qp_record_answers import HOST, TOKEN_URL

from answer_record_helpers import use_loadtest_variables
from query_processing.loadtest.answer_record import read_answer_record
from query_processing.loadtest.qp_record_answers import record_answers_main

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


def test_a_setting_the_load_test_refuses_is_a_usage_error_naming_the_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The old bearer token alone: the settings refuse it as they are built.
    variables = {"QP_LOADTEST_BEARER_TOKEN": "secret-run-token"}
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    never = httpx.MockTransport(pytest.fail)

    argv = ["--host", HOST, "--out", "run"]
    status = record_answers_main(
        argv, client=httpx.Client(transport=never), key_transport=never
    )

    [error] = capsys.readouterr().err.splitlines()
    assert status == 2
    assert error.startswith(
        "qp-record-answers: QP_LOADTEST_BEARER_TOKEN is no longer read"
    )
    assert "secret-run-token" not in error
    assert not list(tmp_path.glob("*_answers.json"))


MARKER = "sk-MARKER-0123456789"


@pytest.mark.parametrize(
    "variables",
    [
        {"QP_LOADTEST_KEY": MARKER},
        {"QP_LOADTEST_KEY__SOURCE": MARKER},
        {"QP_LOADTEST_KEY__SOURCE": "env", "QP_LOADTEST_KEY__VARIABLE": MARKER},
    ],
    ids=["as-the-block", "as-its-source", "as-its-variable"],
)
def test_a_key_typed_in_the_key_block_is_a_usage_error_that_prints_it_not(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    variables: dict[str, str],
) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, variables)
    never = httpx.MockTransport(pytest.fail)

    argv = ["--host", HOST, "--out", "run"]
    status = record_answers_main(
        argv, client=httpx.Client(transport=never), key_transport=never
    )

    printed = capsys.readouterr()
    [error] = printed.err.splitlines()
    assert status == 2
    assert error.startswith("qp-record-answers: QP_LOADTEST_KEY")
    assert MARKER not in printed.out + printed.err


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
