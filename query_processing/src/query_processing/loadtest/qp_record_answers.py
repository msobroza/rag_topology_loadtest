"""``qp-record-answers``: send each request of the load test once, record the answers.

The requests are the load test's request pool, in its order: the questions
file of ``QP_LOADTEST_QUESTIONS_FILE`` when it is set, else the synthetic
questions. Each body is sent once to ``/retrieve`` and then once to
``/query``, one request after the other, with the load test's credentials and
its wait for one answer (``QP_LOADTEST_REQUEST_TIMEOUT_S``). An endpoint whose
load-test weight is 0 (``QP_LOADTEST_RETRIEVE_WEIGHT``,
``QP_LOADTEST_QUERY_WEIGHT``) is not sent: a deployment without a working LLM
is recorded on ``/retrieve`` alone. The answers go to ``<prefix>_answers.json``,
an answer record, which ``qp-compare-answers`` compares with another.

A request the target never answers is retried as ``QP_LOADTEST_RETRIES`` and
``QP_LOADTEST_BACKOFF_S`` say; when it stays unanswered the command stops and
writes nothing, since half a record would pass for a whole one. It runs on
demand, against whatever is deployed: a laptop, Domino, any topology. It is
self-contained: without a questions file it reads none (only the load
test's ``.env``, when there is one), the synthetic questions being built in
process, and calls nothing but its target; the corpus and the models are the
deployment's own.

Example:
    $ qp-record-answers --host http://localhost:8888 --out results/monolith
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path

import httpx

from .answer_record import (
    MalformedAnswerError,
    RecordedAnswer,
    answer_record,
    recorded_answer,
    write_answer_record,
)
from .loadtest_settings import LoadTestSettings
from .request_plans import RequestPool, build_request_pool
from .target_client import RetryPolicy, TargetClient, TargetUnreachableError
from .target_record import TargetPair, target_pair

RETRIEVE_PATH, QUERY_PATH = "/retrieve", "/query"
USAGE_ERROR = 2  # a setting the command cannot run with, as argparse exits
UNREACHABLE = 3  # the target answered no attempt of a request
MALFORMED_ANSWER = 4  # a 200's JSON object was not of the answer's shape


def recorded_endpoints(settings: LoadTestSettings) -> tuple[str, ...]:
    """Return the endpoints each body is sent to: those of a weight above 0.

    The load test's weights say which endpoints a run calls; how large one is
    plays no part here: each body is sent once to each endpoint kept.

    Args:
        settings: The load test's settings, their two weights.

    Returns:
        ``/retrieve`` then ``/query``, each that has a weight above 0.

    Raises:
        ValueError: If both weights are 0; the message names both variables.

    Example:
        >>> recorded_endpoints(LoadTestSettings(query_weight=0))
        ('/retrieve',)
    """
    weights = (
        (RETRIEVE_PATH, settings.retrieve_weight),
        (QUERY_PATH, settings.query_weight),
    )
    endpoints = tuple(path for path, weight in weights if weight > 0)
    if not endpoints:
        raise ValueError(
            "QP_LOADTEST_RETRIEVE_WEIGHT=0 and QP_LOADTEST_QUERY_WEIGHT=0: nothing "
            "to record; expected at least one of them above 0"
        )
    return endpoints


def record_answers(
    poster: TargetClient, pool: RequestPool, endpoints: Sequence[str]
) -> tuple[list[RecordedAnswer], Counter[TargetPair]]:
    """Send each body of ``pool`` once to each of ``endpoints``, in order.

    Args:
        poster: Posts to the target, retrying a request it never answered.
        pool: The load test's request pool; its questions are the bodies.
        endpoints: Where each body goes, in this order.

    Returns:
        The answers in the order sent, and the responses per pair reported.

    Raises:
        TargetUnreachableError: If a request got no answer on any attempt.

    Example:
        >>> answers, seen = record_answers(poster, pool, ("/retrieve", "/query"))
    """
    answers: list[RecordedAnswer] = []
    seen: Counter[TargetPair] = Counter()
    for endpoint, body in _requests(pool, endpoints):
        response = poster.post_json(endpoint, body)
        answers.append(
            recorded_answer(endpoint, body, response.status_code, _json_of(response))
        )
        pair = target_pair(response.headers)
        if pair is not None:
            seen[pair] += 1
    return answers, seen


def _requests(
    pool: RequestPool, endpoints: Sequence[str]
) -> Iterator[tuple[str, dict[str, object]]]:
    for question in pool.questions:
        yield from ((endpoint, dict(question)) for endpoint in endpoints)


def _json_of(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:  # not JSON: recorded with its status alone
        return None


def record_answers_main(
    argv: Sequence[str] | None = None,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run ``qp-record-answers --host <url> --out <prefix>``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        client: What does the HTTP; ``None`` opens one. A test hands it an app
            in process (``fastapi.testclient.TestClient``).
        sleep: The wait between retries.

    Returns:
        0 once the record is written; 2 for settings it cannot run with; 3
        when the target never answered a request, and 4 when a 200 was not
        of the answer's shape, each with nothing written.
    """
    args = _parser().parse_args(argv)
    settings = LoadTestSettings()
    try:
        endpoints = recorded_endpoints(settings)
    except ValueError as exc:
        print(f"qp-record-answers: {exc}", file=sys.stderr)
        return USAGE_ERROR
    # No rerank candidates and no top_n: only a model server's target sends them.
    pool = build_request_pool(
        settings.question_count, 0, 0, questions_file=settings.questions_file
    )
    try:
        answers, seen = _record_with(
            client, args.host, settings, sleep, pool, endpoints
        )
    except (TargetUnreachableError, MalformedAnswerError) as exc:
        print(f"qp-record-answers: {exc}; nothing written", file=sys.stderr)
        unreachable = isinstance(exc, TargetUnreachableError)
        return UNREACHABLE if unreachable else MALFORMED_ANSWER
    path = write_answer_record(args.out, answer_record(answers, seen))
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _record_with(
    client: httpx.Client | None,
    host: str,
    settings: LoadTestSettings,
    sleep: Callable[[float], None],
    pool: RequestPool,
    endpoints: Sequence[str],
) -> tuple[list[RecordedAnswer], Counter[TargetPair]]:
    policy = RetryPolicy(
        settings.request_timeout_s, settings.retries, settings.backoff_s
    )
    # A client handed in is the caller's to close; one opened here is closed here.
    opened = nullcontext(client) if client else httpx.Client(timeout=policy.timeout_s)
    with opened as http:
        poster = TargetClient(http, host, settings.auth_headers, policy, sleep)
        return record_answers(poster, pool, endpoints)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qp-record-answers",
        description="Send each request of the load test once and record the answers.",
    )
    parser.add_argument("--host", required=True, help="the target's base URL")
    parser.add_argument(
        "--out", required=True, type=Path, help="writes <prefix>_answers.json"
    )
    return parser
