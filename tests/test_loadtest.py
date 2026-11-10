"""Tests for ``rag_load_test.loadtest`` helpers, ``rag-compare`` and ``locustfile.py``."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from rag_load_test.loadtest import (
    QUESTIONS,
    RowStats,
    auth_headers,
    compare_main,
    load_stats,
    parse_timings,
    render_markdown,
    stage_events,
    task_weights,
)

LOCUSTFILE = Path(__file__).resolve().parents[1] / "locustfile.py"
STATS_HEADER = (
    "Type,Name,Request Count,Failure Count,Median Response Time,"
    "Average Response Time,Min Response Time,Max Response Time,Average Content Size,"
    "Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%"
).split(",")
BEARER, API_KEY = "RAG_LOADTEST_BEARER_TOKEN", "DOMINO_API_KEY"


# --- locust-free helpers ---


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({BEARER: "tok", API_KEY: "key"}, {"Authorization": "Bearer tok"}),
        ({API_KEY: "key"}, {"X-Domino-Api-Key": "key"}),
        ({BEARER: "", API_KEY: ""}, {}),
        ({}, {}),
    ],
)
def test_auth_headers(env: dict[str, str], expected: dict[str, str]) -> None:
    assert auth_headers(env) == expected


def test_task_weights_defaults_env_override_and_validation() -> None:
    assert task_weights({}) == (1, 3)
    env = {"RAG_LOADTEST_QUERY_WEIGHT": "2", "RAG_LOADTEST_RETRIEVE_WEIGHT": "7"}
    assert task_weights(env) == (2, 7)
    with pytest.raises(ValueError, match="many"):
        task_weights({"RAG_LOADTEST_QUERY_WEIGHT": "many"})


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"timings_ms": {"embed": 4, "total": "9.5"}}, {"embed": 4.0, "total": 9.5}),
        ({}, {}),
        ({"timings_ms": None}, {}),
        ({"timings_ms": "fast"}, {}),
        ({"timings_ms": [1, 2]}, {}),
        ({"timings_ms": {"embed": "fast"}}, {}),
        ({"timings_ms": {"embed": None}}, {}),
    ],
)
def test_parse_timings(body: dict[str, Any], expected: dict[str, float]) -> None:
    assert parse_timings(body) == expected


def test_stage_events_follow_stage_order_and_skip_zero_missing_total() -> None:
    timings = {"total": 20.0, "generate": 8.0, "rerank": 0.0, "embed": 2.0}
    expected = [("/query:embed", 2.0), ("/query:generate", 8.0)]
    assert stage_events("/query", timings) == expected
    assert stage_events("/retrieve", {}) == []


def test_questions_has_fifty_entries() -> None:
    assert len(QUESTIONS) == 50 and all(q for q in QUESTIONS)


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
    for var in ("SHAPE", "QUERY_WEIGHT", "RETRIEVE_WEIGHT"):
        monkeypatch.delenv(f"RAG_LOADTEST_{var}", raising=False)
    return _import_locustfile("locustfile_under_test")


class FakeResponse:
    """Stand-in for locust's ``catch_response`` context; records ``failure`` calls."""

    def __init__(self, status_code: int, body: Any, text: str = "") -> None:
        self.status_code, self._body, self.text = status_code, body, text
        self.failures: list[str] = []

    def json(self) -> Any:
        return self._body

    def failure(self, message: str) -> None:
        self.failures.append(message)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response, self.calls = response, []

    def post(self, path: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((path, kwargs))
        return self.response


def _rag_user(
    module: ModuleType, response: FakeResponse
) -> tuple[Any, list[dict[str, Any]]]:
    """RagUser over fakes; skips ``FastHttpUser.__init__`` (needs a host + session)."""
    fired: list[dict[str, Any]] = []
    request_hook = SimpleNamespace(fire=lambda **event: fired.append(event))
    user = module.RagUser.__new__(module.RagUser)
    user.environment = SimpleNamespace(events=SimpleNamespace(request=request_hook))
    user.client, user.headers = FakeClient(response), {"X-Domino-Api-Key": "key"}
    return user, fired


def test_rag_user_has_both_tasks_with_default_weights(locustfile: ModuleType) -> None:
    user_cls = locustfile.RagUser
    assert (locustfile.QUERY_WEIGHT, locustfile.RETRIEVE_WEIGHT) == (1, 3)
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


def test_ask_fires_stage_events_on_success(locustfile: ModuleType) -> None:
    timings = {"embed": 2.5, "retrieve": 7.0, "rerank": 0.0, "total": 9.5}
    response = FakeResponse(200, {"timings_ms": timings})
    user, fired = _rag_user(locustfile, response)
    user._ask("/retrieve")
    ((path, kwargs),) = user.client.calls
    assert path == "/retrieve"
    assert kwargs["name"] == "/retrieve" and kwargs["catch_response"] is True
    assert kwargs["headers"] == {"X-Domino-Api-Key": "key"}
    assert kwargs["json"]["question"] in QUESTIONS
    assert response.failures == []
    assert [(e["request_type"], e["name"], e["response_time"]) for e in fired] == [
        ("STAGE", "/retrieve:embed", 2.5),
        ("STAGE", "/retrieve:retrieve", 7.0),
    ]
    assert all(e["exception"] is None and e["response_length"] == 0 for e in fired)


def test_ask_marks_failure_on_non_200(locustfile: ModuleType) -> None:
    response = FakeResponse(503, {"detail": "down"}, '{"detail": "embedder down"}')
    user, fired = _rag_user(locustfile, response)
    user._ask("/query")
    assert response.failures == ['HTTP 503: {"detail": "embedder down"}']
    assert fired == []
