"""What a response becomes in the statistics: a failure, or STAGE rows."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.component_requests import GenerateAnswer
from query_processing.entrypoints.qp_serve import create_app
from query_processing.loadtest.response_checks import (
    ResponseVerdict,
    SpentTokens,
    judge_response,
)
from rag_components import FAKE_BACKENDS

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


# --- the tokens of each stage ---

GENERATE_TOKENS = (
    '{"timings_ms": {"embed": 3.0, "generate": 7474.0, "total": 7480.0}, '
    '"tokens": {"generate": {"prompt": 829, "completion": 128}}}'
)
GENERATE_SPENT = (("/query:generate", SpentTokens(829, 128, 7474.0)),)


def test_a_response_with_tokens_becomes_a_verdict_with_its_stages_tokens() -> None:
    verdict = judge_response("/query", 200, GENERATE_TOKENS)
    assert verdict.tokens == GENERATE_SPENT


def test_each_stage_keeps_its_own_tokens_and_its_own_time() -> None:
    # The client knows no stage name: any stage the map holds counts, under the
    # name of its STAGE row, with its own time, never the embed stage's nor the
    # total, and never added to another stage's.
    body = (
        '{"timings_ms": {"embed": 3.0, "draft": 900.0, "answer": 6000.0, '
        '"total": 6910.0}, "tokens": {"draft": {"prompt": 120, "completion": 30}, '
        '"answer": {"prompt": 829, "completion": 128}}}'
    )
    assert judge_response("/query", 200, body).tokens == (
        ("/query:draft", SpentTokens(120, 30, 900.0)),
        ("/query:answer", SpentTokens(829, 128, 6000.0)),
    )


def test_a_retrieve_reports_the_tokens_its_rewrite_spent() -> None:
    body = (
        '{"timings_ms": {"rewrite": 900.0}, '
        '"tokens": {"rewrite": {"prompt": 120, "completion": 30}}}'
    )
    assert judge_response("/retrieve", 200, body).tokens == (
        ("/retrieve:rewrite", SpentTokens(120, 30, 900.0)),
    )


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
        # the total is no stage: it has no STAGE row
        '{"timings_ms": {"total": 7474.0}, '
        '"tokens": {"total": {"prompt": 829, "completion": 128}}}',
    ],
)
def test_a_response_without_usable_tokens_has_none(body: str) -> None:
    assert judge_response("/query", 200, body).tokens == ()


@pytest.mark.parametrize(
    ("rewrite_ms", "rewrite_counts"),
    [
        ("", '{"prompt": 1, "completion": 1}'),  # no time
        ('"rewrite": 0, ', '{"prompt": 1, "completion": 1}'),  # timed at zero
        ('"rewrite": null, ', '{"prompt": 1, "completion": 1}'),  # no number
        ('"rewrite": "x", ', '{"prompt": 1, "completion": 1}'),
        ('"rewrite": true, ', '{"prompt": 1, "completion": 1}'),
        ('"rewrite": 900.0, ', '{"prompt": 1}'),  # no completion
        ('"rewrite": 900.0, ', '{"prompt": 1, "completion": "few"}'),
    ],
    ids=[
        "no-time",
        "zero-time",
        "null-time",
        "text-time",
        "true-time",
        "no-completion",
        "unusable-count",
    ],
)
def test_a_stage_with_unusable_tokens_is_left_out_alone(
    rewrite_ms: str, rewrite_counts: str
) -> None:
    body = (
        f'{{"timings_ms": {{{rewrite_ms}"generate": 7474.0}}, '
        f'"tokens": {{"rewrite": {rewrite_counts}, '
        '"generate": {"prompt": 829, "completion": 128}}}'
    )
    verdict = judge_response("/query", 200, body)
    assert verdict.tokens == GENERATE_SPENT
    # The stage whose time is unusable has no row either; the others keep theirs.
    assert ("/query:generate", 7474.0) in verdict.stage_rows


def test_a_failure_has_no_tokens() -> None:
    assert judge_response("/query", 503, GENERATE_TOKENS).tokens == ()


class FixedTokensLLM(FakeLLM):
    """The fake LLM, whose every answer reports 829 prompt and 128 completion tokens."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        return GenerateAnswer(text="Paris.", prompt_tokens=829, completion_tokens=128)


def test_the_tokens_the_workflow_app_writes_are_the_tokens_the_load_test_reads(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # The keys are written by the workflow app and read in
    # loadtest/response_checks.py; a body the app built proves they agree.
    settings = settings_factory(**FAKE_BACKENDS)
    root = CompositionRoot(settings, llm=FixedTokensLLM())
    with TestClient(create_app(root)) as app:
        response = app.post("/query", json={"question": "Capital of France?"})
    assert response.status_code == 200, response.text

    verdict = judge_response("/query", response.status_code, response.text)

    generate_ms = response.json()["timings_ms"]["generate"]
    assert generate_ms > 0
    assert verdict.tokens == (("/query:generate", SpentTokens(829, 128, generate_ms)),)
