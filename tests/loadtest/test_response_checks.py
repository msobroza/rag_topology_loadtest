"""What a response becomes in the statistics: a failure, or STAGE rows."""

from __future__ import annotations

import pytest

from rag_load_test.loadtest.response_checks import (
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
        # STAGES order, zero and missing stages skipped, ``total`` never a row
        (
            "/query",
            200,
            OK_TIMINGS,
            _stage_rows(("/query:embed", 2.0), ("/query:generate", 8.0)),
        ),
        (
            "/retrieve",
            200,
            '{"timings_ms": {"embed": 4, "total": "9.5", "retrieve": "1.5"}}',
            _stage_rows(("/retrieve:embed", 4.0), ("/retrieve:retrieve", 1.5)),
        ),
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


def test_a_response_with_tokens_becomes_a_verdict_with_tokens() -> None:
    body = '{"timings_ms": {"generate": 7474.0}, "tokens": {"prompt": 829, "completion": 128}}'
    assert judge_response("/query", 200, body).tokens == AnswerTokens(829, 128, 7474.0)


@pytest.mark.parametrize(
    "body",
    [
        '{"timings_ms": {"generate": 7474.0}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {"prompt": "many", "completion": 128}}',
        '{"timings_ms": {}, "tokens": {"prompt": 829, "completion": 128}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {"prompt": 829}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {"prompt": 8.5, "completion": 1}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": {"prompt": true, "completion": 1}}',
        '{"timings_ms": {"generate": 7474.0}, "tokens": [829, 128]}',
        '{"timings_ms": {"generate": 0}, "tokens": {"prompt": 829, "completion": 128}}',
    ],
)
def test_a_response_without_usable_tokens_has_none(body: str) -> None:
    assert judge_response("/query", 200, body).tokens is None


def test_a_failure_has_no_tokens() -> None:
    body = '{"timings_ms": {"generate": 7474.0}, "tokens": {"prompt": 829, "completion": 128}}'
    assert judge_response("/query", 503, body).tokens is None
