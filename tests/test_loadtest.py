"""Tests for ``rag_load_test.loadtest`` request plans, ``rag-compare`` and ``locustfile.py``."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
from rag_load_test.loadtest.run_comparison import (
    RowStats,
    compare_main,
    load_stats,
    render_markdown,
)

LOCUSTFILE = Path(__file__).resolve().parents[1] / "locustfile.py"
STATS_HEADER = (
    "Type,Name,Request Count,Failure Count,Median Response Time,"
    "Average Response Time,Min Response Time,Max Response Time,Average Content Size,"
    "Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%"
).split(",")
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

    assert len(pool.questions) == 9 and all(pool.questions)
    assert len(pool.rerank_documents) == 7 and all(pool.rerank_documents)
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
    assert verdict.failure is not None and verdict.failure.startswith(failure)
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


# --- rag-compare ---

_ROW_KEYS = ("Type", "Name", "Request Count", "Failure Count", "Requests/s")
MONOLITH = (
    ("POST", "/query", 100, 2, 4.5, 20, 30, 40),
    ("STAGE", "/query:embed", 98, 0, 4.4, 3, 5, 6),
    ("", "Aggregated", 198, 2, 8.9, 10, 30, 40),
)
SPLIT = (
    ("POST", "/query", 100, 0, 5.0, 25, 45, 60),
    ("", "Aggregated", 100, 0, 5.0, 25, 45, 60),
)


def _stats_csv(prefix: Path, *rows: tuple[Any, ...]) -> Path:
    """Write ``<prefix>_stats.csv``; a row is (Type, Name, requests, failures, rps, p50, p95, p99)."""
    with Path(f"{prefix}_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_HEADER)
        writer.writeheader()
        for row in rows:
            keys = (*_ROW_KEYS, "50%", "95%", "99%")
            cells = dict(zip(keys, map(str, row), strict=True))
            writer.writerow({**dict.fromkeys(STATS_HEADER, "0"), **cells})
    return prefix


def test_load_stats_keys_rows_and_zeroes_na_percentiles(tmp_path: Path) -> None:
    idle = ("POST", "/retrieve", 0, 0, 0.0, "N/A", "N/A", "N/A")
    stats = load_stats(_stats_csv(tmp_path / "monolith", *MONOLITH, idle))
    expected_keys = {
        "POST /query",
        "STAGE /query:embed",
        "POST /retrieve",
        "Aggregated",
    }
    assert set(stats) == expected_keys
    assert stats["POST /query"] == RowStats(100, 2, 4.5, 20.0, 30.0, 40.0)
    assert stats["Aggregated"].requests == 198
    assert stats["POST /retrieve"] == RowStats(0, 0, 0.0, 0.0, 0.0, 0.0)


def test_render_markdown_orders_tables_and_computes_delta_p95(tmp_path: Path) -> None:
    runs = {
        "monolith": load_stats(_stats_csv(tmp_path / "monolith", *MONOLITH)),
        "split-all": load_stats(_stats_csv(tmp_path / "split-all", *SPLIT)),
    }
    markdown = render_markdown(runs)
    headings = [line for line in markdown.splitlines() if line.startswith("### ")]
    assert headings == ["### POST /query", "### STAGE /query:embed", "### Aggregated"]
    assert "Δp95 is relative to `monolith`" in markdown
    assert "| monolith | 100 | 2.0% | 4.5 | 20 | 30 | 40 | — |" in markdown
    split_row = "| split-all | 100 | 0.0% | 5.0 | 25 | 45 | 60 | +15.0 ms (+50.0%) |"
    assert split_row in markdown
    assert "| split-all | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in markdown


def test_compare_main_writes_out_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monolith = _stats_csv(tmp_path / "monolith", *MONOLITH)
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    out = tmp_path / "comparison.md"
    assert compare_main([str(monolith), str(split), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and f"wrote {out}" in captured.err
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Locust run comparison")
    assert "Runs: monolith, split-all." in text and "+15.0 ms (+50.0%)" in text


def _target_json(prefix: Path, target: str, *seen: tuple[str, str, int]) -> None:
    """Write ``<prefix>_target.json``; a sighting is (topology, run label, responses)."""
    write_target_record(str(prefix), target, Counter({(t, r): n for t, r, n in seen}))


def test_compare_names_topology_and_run_label_from_target_records(
    tmp_path: Path,
) -> None:
    monolith = _stats_csv(tmp_path / "monolith", *MONOLITH)
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    _target_json(monolith, "rag", ("monolith", "monolith-cpu", 198))
    out = tmp_path / "comparison.md"
    assert compare_main([str(monolith), str(split), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    intro = (
        "Runs: monolith (topology `monolith`, run label `monolith-cpu`), split-all. "
        "Δp95 is relative to `monolith`."
    )
    assert intro in text.splitlines()
    assert "Warning" not in text


def test_compare_warns_when_a_run_saw_more_than_one_pair(tmp_path: Path) -> None:
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    _target_json(split, "rag", ("split-all", "a", 60), ("split-reranker", "b", 40))
    markdown = render_markdown(
        {"split-all": load_stats(split)},
        {"split-all": _read_json(Path(f"{split}_target.json"))},
    )
    intro = (
        "Runs: split-all (topology `split-all`, run label `a`; "
        "topology `split-reranker`, run label `b`). Δp95 is relative to `split-all`."
    )
    lines = markdown.splitlines()
    assert intro in lines
    warning = (
        "> **Warning:** run `split-all` saw 2 topology/run label pairs; "
        "its numbers mix configurations."
    )
    assert lines[lines.index(intro) + 2] == warning


def test_compare_lists_a_model_server_run_by_name_alone(tmp_path: Path) -> None:
    ovms = _stats_csv(tmp_path / "ovms-rerank", *SPLIT)
    _target_json(ovms, "rerank")
    out = tmp_path / "comparison.md"
    assert compare_main([str(ovms), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Runs: ovms-rerank. Δp95 is relative to `ovms-rerank`." in text.splitlines()


# --- locustfile.py ---


def _import_locustfile(name: str) -> ModuleType:
    """Execute ``locustfile.py`` under ``name``; its shape class depends on the env."""
    spec = importlib.util.spec_from_file_location(name, LOCUSTFILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def locustfile(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for var in ("TARGET", "SHAPE", "QUERY_WEIGHT", "RETRIEVE_WEIGHT", "QUESTIONS"):
        monkeypatch.delenv(f"RAG_LOADTEST_{var}", raising=False)
    return _import_locustfile("locustfile_under_test")


def test_locustfile_pool_follows_the_settings_and_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_TOP_K_RETRIEVE", "7")
    monkeypatch.setenv("RAG_TOP_K_RERANK", "3")
    monkeypatch.setenv("RAG_LOADTEST_QUESTIONS", "9")

    pool = _import_locustfile("locustfile_sized_pool").POOL

    assert (len(pool.questions), len(pool.rerank_documents), pool.top_n) == (9, 7, 3)


def test_rag_user_has_both_tasks_with_default_weights(locustfile: ModuleType) -> None:
    user_cls = locustfile.RagUser
    assert user_cls.query.locust_task_weight == 1
    assert user_cls.retrieve.locust_task_weight == 3
    names = sorted(t.__name__ for t in user_cls.tasks)
    assert names == ["query", "retrieve", "retrieve", "retrieve"]
    assert not hasattr(locustfile, "StepLoadShape")


def test_step_shape_ramps_then_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOADTEST_SHAPE", "step")
    shape = _import_locustfile("locustfile_step_shape").StepLoadShape()
    for run_time, expected in ((0.0, (5, 5.0)), (239.0, (40, 40.0)), (240.0, None)):
        monkeypatch.setattr(shape, "get_run_time", lambda t=run_time: t)
        assert shape.tick() == expected


@pytest.mark.parametrize(
    ("target", "active"),
    [("rag", "RagUser"), ("rerank", "RerankUser"), ("embeddings", "EmbeddingsUser")],
)
def test_loadtest_target_selects_one_user_class(
    monkeypatch: pytest.MonkeyPatch, target: str, active: str
) -> None:
    monkeypatch.setenv("RAG_LOADTEST_TARGET", target)
    module = _import_locustfile(f"locustfile_target_{target}")
    flags = {
        name: getattr(module, name).abstract
        for name in ("RagUser", "RerankUser", "EmbeddingsUser")
    }
    assert [name for name, abstract in flags.items() if not abstract] == [active]


class FakeCaughtResponse:
    """Stand-in for Locust's ``catch_response`` context: one canned response."""

    def __init__(
        self, status_code: int, text: str, headers: dict[str, str] | None = None
    ) -> None:
        self.status_code, self.text, self.headers = status_code, text, headers or {}
        self.failures: list[str] = []
        self.successes = 0

    def failure(self, message: str) -> None:
        self.failures.append(message)

    def success(self) -> None:
        self.successes += 1

    def __enter__(self) -> FakeCaughtResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeLocustUser:
    """What ``send`` touches on a user: ``client.post`` and the request event."""

    def __init__(self, response: FakeCaughtResponse) -> None:
        self.posted: list[dict[str, Any]] = []
        self.fired: list[dict[str, Any]] = []
        self._response = response
        self.client = SimpleNamespace(post=self._post)
        request_event = SimpleNamespace(fire=lambda **event: self.fired.append(event))
        self.environment = SimpleNamespace(
            events=SimpleNamespace(request=request_event)
        )

    def _post(self, path: str, **kwargs: Any) -> FakeCaughtResponse:
        self.posted.append({"path": path, **kwargs})
        return self._response


def test_send_records_a_success_its_stage_rows_and_the_target_pair(
    locustfile: ModuleType,
) -> None:
    headers = {"X-RAG-Topology": "monolith", "X-RAG-Run-Label": "mono"}
    body = '{"timings_ms": {"embed": 2.5, "rerank": 0, "total": 9}}'
    response = FakeCaughtResponse(200, body, headers)
    user = FakeLocustUser(response)

    locustfile.send(user, RequestPlan("/retrieve", {"question": "q"}, "/retrieve"))

    [posted] = user.posted
    assert (posted["path"], posted["json"]) == ("/retrieve", {"question": "q"})
    assert (posted["name"], posted["catch_response"]) == ("/retrieve", True)
    assert (response.successes, response.failures) == (1, [])
    fired = [(e["request_type"], e["name"], e["response_time"]) for e in user.fired]
    assert fired == [("STAGE", "/retrieve:embed", 2.5)]
    assert locustfile.RESPONSES_SEEN == {("monolith", "mono"): 1}


def test_send_marks_a_non_json_200_as_a_failure_and_fires_nothing(
    locustfile: ModuleType,
) -> None:
    response = FakeCaughtResponse(200, "<html>proxy error</html>")
    user = FakeLocustUser(response)

    locustfile.send(user, RequestPlan("/query", {"question": "q"}, "/query"))

    assert response.successes == 0
    assert response.failures == [
        "HTTP 200 but the body is not a JSON object: <html>proxy error</html>"
    ]
    assert user.fired == []


def test_stop_listener_writes_the_target_record_beside_the_csv(
    locustfile: ModuleType, tmp_path: Path
) -> None:
    locustfile.RESPONSES_SEEN.update({("monolith", "mono"): 2})
    options = SimpleNamespace(csv_prefix=str(tmp_path / "monolith"))
    locustfile.write_target_record_on_stop(SimpleNamespace(parsed_options=options))
    assert _read_json(tmp_path / "monolith_target.json") == {
        "target": "rag",
        "seen": [{"topology": "monolith", "run_label": "mono", "responses": 2}],
    }
    locustfile.write_target_record_on_stop(SimpleNamespace(parsed_options=None))
