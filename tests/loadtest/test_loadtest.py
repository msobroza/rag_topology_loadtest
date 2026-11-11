"""Tests for ``rag_load_test.loadtest.request_plans``: config, request plans, verdicts, target record."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rag_load_test.loadtest.request_plans import (
    LoadTestConfig,
    RequestPlan,
    RequestPool,
    ResponseVerdict,
    build_request_pool,
    judge_response,
    plan_embeddings_request,
    plan_rag_request,
    plan_rerank_request,
    read_loadtest_config,
    target_pair,
    write_target_record,
)
from repo_paths import REPO_ROOT

LOCUSTFILE = REPO_ROOT / "locustfile.py"
STATS_HEADER = [
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Median Response Time",
    "Average Response Time",
    "Min Response Time",
    "Max Response Time",
    "Average Content Size",
    "Requests/s",
    "Failures/s",
    "50%",
    "66%",
    "75%",
    "80%",
    "90%",
    "95%",
    "98%",
    "99%",
    "99.9%",
    "99.99%",
    "100%",
]
BEARER, API_KEY = "RAG_LOADTEST_BEARER_TOKEN", "DOMINO_API_KEY"


# --- locust-free helpers ---


def test_loadtest_config_defaults() -> None:
    assert read_loadtest_config({}) == LoadTestConfig(
        target="rag",
        query_weight=1,
        retrieve_weight=3,
        step_shape=False,
        auth_headers={},
        question_count=50,
    )


def test_loadtest_config_reads_every_variable() -> None:
    env = {
        "RAG_LOADTEST_TARGET": "rerank",
        "RAG_LOADTEST_QUERY_WEIGHT": "2",
        "RAG_LOADTEST_RETRIEVE_WEIGHT": "7",
        "RAG_LOADTEST_SHAPE": "step",
        "RAG_LOADTEST_QUESTIONS": "12",
        API_KEY: "key",
    }
    config = read_loadtest_config(env)
    assert (config.target, config.query_weight, config.retrieve_weight) == (
        "rerank",
        2,
        7,
    )
    assert config.step_shape is True
    assert config.auth_headers == {"X-Domino-Api-Key": "key"}
    assert config.question_count == 12


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({BEARER: "tok", API_KEY: "key"}, {"Authorization": "Bearer tok"}),
        ({API_KEY: "key"}, {"X-Domino-Api-Key": "key"}),
        ({BEARER: "", API_KEY: ""}, {}),
        ({}, {}),
    ],
)
def test_loadtest_config_auth_headers_bearer_wins(
    env: dict[str, str], expected: dict[str, str]
) -> None:
    assert read_loadtest_config(env).auth_headers == expected


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RAG_LOADTEST_TARGET": "ovms"}, r"'ovms'.*rag, rerank, embeddings"),
        ({"RAG_LOADTEST_QUERY_WEIGHT": "many"}, r"RAG_LOADTEST_QUERY_WEIGHT.*'many'"),
        ({"RAG_LOADTEST_RETRIEVE_WEIGHT": "1.5"}, r"RETRIEVE_WEIGHT.*'1\.5'"),
        ({"RAG_LOADTEST_QUESTIONS": "lots"}, r"RAG_LOADTEST_QUESTIONS.*'lots'"),
        ({"RAG_LOADTEST_QUESTIONS": "0"}, r"RAG_LOADTEST_QUESTIONS.*'0'.*at least 1"),
    ],
)
def test_loadtest_config_rejects_bad_values(env: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_loadtest_config(env)


# --- what to send ---


POOL = RequestPool(
    questions=("q-one", "q-two", "q-three"),
    rerank_documents=("doc a", "doc b", "doc c", "doc d"),
    top_n=2,
)


def test_build_request_pool_sizes_questions_and_candidates() -> None:
    pool = build_request_pool(question_count=9, candidates=7, top_n=3)

    assert len(pool.questions) == 9
    assert all(pool.questions)
    assert len(pool.rerank_documents) == 7
    assert all(pool.rerank_documents)
    assert pool.top_n == 3


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_plan_rag_request_asks_one_seeded_question(endpoint: str) -> None:
    plan = plan_rag_request(endpoint, POOL, random.Random(7))
    question = random.Random(7).choice(POOL.questions)
    assert plan == RequestPlan(endpoint, {"question": question}, endpoint)


def test_plan_rerank_request_matches_the_workflow_rerank_stage() -> None:
    plan = plan_rerank_request("bge-reranker-base", POOL, random.Random(7))
    assert (plan.path, plan.stats_name) == ("/v3/rerank", "/v3/rerank")
    assert plan.payload == {
        "model": "bge-reranker-base",
        "query": random.Random(7).choice(POOL.questions),
        "documents": ["doc a", "doc b", "doc c", "doc d"],
        "top_n": 2,
    }


def test_plan_embeddings_request_sends_one_question_in_a_list() -> None:
    plan = plan_embeddings_request("bge-small-en-v1.5", POOL, random.Random(7))
    assert (plan.path, plan.stats_name) == ("/v3/embeddings", "/v3/embeddings")
    assert plan.payload == {
        "model": "bge-small-en-v1.5",
        "input": [random.Random(7).choice(POOL.questions)],
    }


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


def test_default_pool_has_fifty_questions() -> None:
    pool = build_request_pool(read_loadtest_config({}).question_count, 20, 5)
    assert len(pool.questions) == 50


# --- target record ---

TOPOLOGY, RUN_LABEL = "X-RAG-Topology", "X-RAG-Run-Label"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({TOPOLOGY: "monolith", RUN_LABEL: "mono-a"}, ("monolith", "mono-a")),
        ({TOPOLOGY: "monolith"}, ("monolith", "")),
        ({RUN_LABEL: "mono-a"}, ("", "mono-a")),
        # Neither header: not the workflow app answering (a proxy's error page, a
        # model server, a connection error), so there is nothing to count.
        ({"Content-Type": "text/html"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_target_pair(
    headers: dict[str, str] | None, expected: tuple[str, str] | None
) -> None:
    assert target_pair(headers) == expected


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_target_record_puts_most_responses_first(tmp_path: Path) -> None:
    counts = Counter(
        {("monolith", "b"): 5, ("split-all", "x"): 9, ("monolith", "a"): 5}
    )
    write_target_record(str(tmp_path / "monolith"), "rag", counts)
    assert _read_json(tmp_path / "monolith_target.json") == {
        "target": "rag",
        "seen": [
            {"topology": "split-all", "run_label": "x", "responses": 9},
            {"topology": "monolith", "run_label": "a", "responses": 5},
            {"topology": "monolith", "run_label": "b", "responses": 5},
        ],
    }


@pytest.mark.parametrize("target", ["rerank", "embeddings"])
def test_write_target_record_model_server_sees_nothing(
    tmp_path: Path, target: str
) -> None:
    write_target_record(str(tmp_path / "ovms"), target, Counter({("", ""): 40}))
    assert _read_json(tmp_path / "ovms_target.json") == {"target": target, "seen": []}


@pytest.mark.parametrize("csv_prefix", [None, ""])
def test_write_target_record_without_csv_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, csv_prefix: str | None
) -> None:
    monkeypatch.chdir(tmp_path)
    write_target_record(csv_prefix, "rag", Counter({("m", "m"): 1}))
    assert list(tmp_path.iterdir()) == []
