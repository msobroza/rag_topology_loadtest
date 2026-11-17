"""What a response becomes in the statistics: a failure, or STAGE rows."""

from __future__ import annotations

import pytest

from query_processing.entrypoints.qp_serve import QueryResponse
from query_processing.loadtest.response_checks import (
    AnswerTokens,
    ResponseVerdict,
    judge_response,
)

# --- what to record ---

OK_TIMINGS = '{"timings_ms": {"total": 20, "generate": 8, "rerank": 0, "embed": 2}}'


def _stage_rows(*rows: tuple[str, float]) -> ResponseVerdict:
    """A success verdict firing these ``(row name, ms)`` STAGE rows."""
    return ResponseVerdict(failure=None, stage_rows=rows)


NO_ROWS = ResponseVerdict(failure=None, stage_rows=())
NOT_AN_OBJECT = "HTTP 200 but the body is not a JSON object: "


@pytest.mark.parametrize(
    ("endpoint", "status", "body", "expected"),
    [
        # the response's order, a stage timed at zero skipped, ``total`` never a row
        (
            "/query",
            200,
            OK_TIMINGS,
            _stage_rows(("/query:generate", 8.0), ("/query:embed", 2.0)),
        ),
        (
            "/retrieve",
            200,
            '{"timings_ms": {"embed": 4, "total": "9.5", "retrieve": "1.5"}}',
            _stage_rows(("/retrieve:embed", 4.0), ("/retrieve:retrieve", 1.5)),
        ),
        # any stage name the response gives is a row: the client knows no stages
        (
            "/query",
            200,
            '{"timings_ms": {"route": 1.5, "lookup": 3, "total": 5}}',
            _stage_rows(("/query:route", 1.5), ("/query:lookup", 3.0)),
        ),
        ("/query", 200, '{"timings_ms": {"total": 5}}', NO_ROWS),
        # a missing or malformed ``timings_ms`` is still a success, with no rows
        ("/query", 200, "{}", NO_ROWS),
        ("/query", 200, '{"timings_ms": null}', NO_ROWS),
        ("/query", 200, '{"timings_ms": "fast"}', NO_ROWS),
        ("/query", 200, '{"timings_ms": [1, 2]}', NO_ROWS),
        ("/query", 200, '{"timings_ms": {"embed": "fast"}}', NO_ROWS),
        ("/query", 200, '{"timings_ms": {"embed": null}}', NO_ROWS),
        ("/v3/rerank", 200, '{"results": []}', NO_ROWS),
    ],
)
def test_judge_response_success(
    endpoint: str, status: int, body: str, expected: ResponseVerdict
) -> None:
    assert judge_response(endpoint, status, body) == expected


@pytest.mark.parametrize(
    ("status", "body", "failure"),
    [
        (503, '{"detail": "embedder down"}', 'HTTP 503: {"detail": "embedder down"}'),
        (500, "x" * 300, "HTTP 500: " + "x" * 200),
        (404, "", "HTTP 404: "),
        (200, "<html>proxy login</html>", NOT_AN_OBJECT),
        (200, "", NOT_AN_OBJECT),
        (200, "[1, 2]", NOT_AN_OBJECT),
        (200, '"ok"', NOT_AN_OBJECT),
    ],
)
def test_judge_response_failure(status: int, body: str, failure: str) -> None:
    verdict = judge_response("/query", status, body)
    assert verdict.failure is not None
    assert verdict.failure.startswith(failure)
    assert verdict.stage_rows == ()
    assert body[:200] in verdict.failure


# --- the tokens of an answer ---

GENERATE_TOKENS = (
    '{"timings_ms": {"embed": 3.0, "generate": 7474.0, "total": 7480.0}, '
    '"tokens": {"generate": {"prompt": 829, "completion": 128}}}'
)


def test_a_response_with_tokens_becomes_a_verdict_with_tokens() -> None:
    verdict = judge_response("/query", 200, GENERATE_TOKENS)
    assert verdict.tokens == AnswerTokens(829, 128, 7474.0)


def test_the_tokens_of_every_stage_add_up_with_the_time_of_those_stages() -> None:
    # The client knows no stage name: any stage the map holds counts, and only
    # the time of those stages, never the embed stage's nor the total.
    body = (
        '{"timings_ms": {"embed": 3.0, "draft": 900.0, "answer": 6000.0, '
        '"total": 6910.0}, "tokens": {"draft": {"prompt": 120, "completion": 30}, '
        '"answer": {"prompt": 829, "completion": 128}}}'
    )
    assert judge_response("/query", 200, body).tokens == AnswerTokens(949, 158, 6900.0)


def test_a_retrieve_is_no_answer_even_when_its_rewrite_spent_tokens() -> None:
    # Main's table counts answers, which only /query gives: a /retrieve's
    # rewrite tokens would lower every per-answer figure.
    body = (
        '{"timings_ms": {"rewrite": 900.0}, '
        '"tokens": {"rewrite": {"prompt": 120, "completion": 30}}}'
    )
    assert judge_response("/retrieve", 200, body).tokens is None


@pytest.mark.parametrize(
    "body",
    [
        '{"timings_ms": {"generate": 7474.0}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {}}',
        '{"timings_ms": {"generate": 7474.0}, '
        '"tokens": {"generate": {"prompt": "many", "completion": 128}}}',
        '{"timings_ms": {}, "tokens": {"generate": {"prompt": 829, "completion": 128}}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {"generate": {"prompt": 829}}}',
        '{"timings_ms": {"generate": 7474.0}, '
        '"tokens": {"generate": {"prompt": 8.5, "completion": 1}}}',
        '{"timings_ms": {"generate": 7474.0}, '
        '"tokens": {"generate": {"prompt": true, "completion": 1}}}',
        '{"timings_ms": {"generate": 7474.0}, '
        '"tokens": {"generate": {"prompt": -1, "completion": 1}}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {"generate": [829, 128]}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": [829, 128]}',
        '{"timings_ms": {"generate": 0}, '
        '"tokens": {"generate": {"prompt": 829, "completion": 128}}}',
        '{"timings_ms": {"generate": "slow"}, '
        '"tokens": {"generate": {"prompt": 829, "completion": 128}}}',
        '{"timings_ms": "fast", '
        '"tokens": {"generate": {"prompt": 829, "completion": 128}}}',
        # one stage of two without its time: the sum would credit it with none
        '{"timings_ms": {"generate": 7474.0}, '
        '"tokens": {"rewrite": {"prompt": 1, "completion": 1}, '
        '"generate": {"prompt": 829, "completion": 128}}}',
    ],
)
def test_a_response_without_usable_tokens_has_none(body: str) -> None:
    assert judge_response("/query", 200, body).tokens is None


def test_a_failure_has_no_tokens() -> None:
    assert judge_response("/query", 503, GENERATE_TOKENS).tokens is None


def test_the_tokens_the_workflow_app_writes_are_the_tokens_the_load_test_reads() -> (
    None
):
    # The keys are written by the workflow app and read in
    # loadtest/response_checks.py; a body qp-serve would send proves they agree.
    served = QueryResponse(
        request_id="r-1",
        topology="monolith",
        run_label="monolith",
        question="What is the capital of France?",
        mode="query",
        answer="Paris.",
        passages=[],
        timings_ms={"embed": 3.0, "generate": 7474.0, "total": 7480.0},
        tokens={"generate": {"prompt": 829, "completion": 128}},
        metadata_filter=[],
    )

    verdict = judge_response("/query", 200, served.model_dump_json())

    assert verdict.tokens == AnswerTokens(829, 128, 7474.0)
