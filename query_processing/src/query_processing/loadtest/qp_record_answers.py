"""``qp-record-answers``: send each request of the load test once, record the answers.

The requests are the load test's request pool, in its order: the questions
file of ``QP_LOADTEST_QUESTIONS_FILE`` when it is set, else the synthetic
questions. Each body is sent once to ``/retrieve`` and then once to
``/query``, one request after the other, with the load test's API key and its
wait for one answer (``QP_LOADTEST_REQUEST_TIMEOUT_S``). An endpoint whose
load-test weight is 0 (``QP_LOADTEST_RETRIEVE_WEIGHT``,
``QP_LOADTEST_QUERY_WEIGHT``) is not sent: a deployment without a working LLM
is recorded on ``/retrieve`` alone. The answers go to ``<prefix>_answers.json``,
an answer record, which ``qp-compare-answers`` compares with another.

A request the target never answers is retried as ``QP_LOADTEST_RETRIES`` and
``QP_LOADTEST_BACKOFF_S`` say; when it stays unanswered the command stops and
writes nothing, since half a record would pass for a whole one. A 401 tells
the key source of the load test's ``key`` block and is sent again within the
same retries; once they run out, its response is recorded as any other. A
``key`` block start-up would refuse stops it before any request. It runs on
demand, against whatever is deployed: a laptop, Domino, any topology. It is
self-contained: without a questions file it reads none (only the load
test's ``.env``, when there is one), the synthetic questions being built in
process, and calls nothing but its target and, with the ``domino`` key source,
the run's token URL; the corpus and the models are the deployment's own.

Example:
    $ qp-record-answers --host http://localhost:8888 --out results/monolith
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path

import httpx

from ..domain.errors import RagDependencyError
from .answer_record import (
    MalformedAnswerError,
    RecordedAnswer,
    answer_record,
    recorded_answer,
    write_answer_record,
)
from .loadtest_settings import LoadTestSettings
from .request_plans import RequestPool, build_request_pool
from .target_client import (
    RetryPolicy,
    TargetClient,
    TargetKey,
    TargetUnreachableError,
    opened_target_key,
)
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
    key_transport: httpx.BaseTransport | None = None,
) -> int:
    """Run ``qp-record-answers --host <url> --out <prefix>``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        client: What does the HTTP; ``None`` opens one. A test hands it an app
            in process (``fastapi.testclient.TestClient``).
        sleep: The wait between retries.
        key_transport: What the ``domino`` key source asks its token URL
            through; ``None`` for the network. A test hands it a stand-in.

    Returns:
        0 once the record is written; 2 for settings it cannot run with, a
        ``key`` block start-up would refuse included; 3 when the target never
        answered a request or the key source could not be asked, and 4 when a
        200 was not of the answer's shape, each with nothing written.
    """
    args = _parser().parse_args(argv)
    with ExitStack() as opened:
        try:  # a setting refused as the settings are built is a usage error too
            settings = LoadTestSettings()
            endpoints = recorded_endpoints(settings)
            key = opened.enter_context(opened_target_key(settings, key_transport))
        except ValueError as exc:  # a MissingKeyError too
            print(f"qp-record-answers: {exc}", file=sys.stderr)
            return USAGE_ERROR
        poster = _poster(client, args.host, settings, key, sleep, opened)
        return _record_and_write(poster, settings, endpoints, args.out)


def _record_and_write(
    poster: TargetClient,
    settings: LoadTestSettings,
    endpoints: Sequence[str],
    out: Path,
) -> int:
    # No rerank candidates and no top_n: only a model server's target sends them.
    pool = build_request_pool(
        settings.question_count, 0, 0, questions_file=settings.questions_file
    )
    try:
        answers, seen = record_answers(poster, pool, endpoints)
    except (TargetUnreachableError, RagDependencyError, MalformedAnswerError) as exc:
        print(f"qp-record-answers: {exc}; nothing written", file=sys.stderr)
        malformed = isinstance(exc, MalformedAnswerError)
        return MALFORMED_ANSWER if malformed else UNREACHABLE
    path = write_answer_record(out, answer_record(answers, seen))
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _poster(
    client: httpx.Client | None,
    host: str,
    settings: LoadTestSettings,
    key: TargetKey,
    sleep: Callable[[float], None],
    opened: ExitStack,
) -> TargetClient:
    policy = RetryPolicy(
        settings.request_timeout_s, settings.retries, settings.backoff_s
    )
    # A client handed in is the caller's to close; one opened here is closed here.
    if client is None:
        client = opened.enter_context(httpx.Client(timeout=policy.timeout_s))
    return TargetClient(client, host, key, policy, sleep)


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
