"""Tests for the top-level ``locustfile.py``: its user classes, load shape and ``send``."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from locust.stats import PERCENTILES_TO_REPORT, RequestStats, StatsCSV

from loadtest_settings_helpers import clear_loadtest_variables
from rag_load_test.loadtest.rag_compare import load_stats
from rag_load_test.loadtest.request_plans import RequestPlan
from repo_paths import REPO_ROOT

LOCUSTFILE = REPO_ROOT / "locustfile.py"


# --- locustfile.py ---


def _import_locustfile(name: str) -> ModuleType:
    """Execute ``locustfile.py`` under ``name``; its shape class depends on the env."""
    spec = importlib.util.spec_from_file_location(name, LOCUSTFILE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_loadtest_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_loadtest_variables(monkeypatch)


@pytest.fixture
def locustfile() -> ModuleType:
    return _import_locustfile("locustfile_under_test")


def test_locustfile_pool_follows_the_settings(
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


def test_step_shape_follows_its_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOADTEST_SHAPE", "step")
    monkeypatch.setenv("RAG_LOADTEST_STEP_USERS", "2,4")
    monkeypatch.setenv("RAG_LOADTEST_STEP_SECONDS", "10")
    shape = _import_locustfile("locustfile_step_settings").StepLoadShape()
    for run_time, expected in ((9.0, (2, 2.0)), (10.0, (4, 4.0)), (20.0, None)):
        monkeypatch.setattr(shape, "get_run_time", lambda t=run_time: t)
        assert shape.tick() == expected


def test_every_user_waits_as_the_settings_say(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOADTEST_WAIT_MIN_S", "0.5")
    monkeypatch.setenv("RAG_LOADTEST_WAIT_MAX_S", "0.5")
    module = _import_locustfile("locustfile_wait_time")
    for name in ("RagUser", "RerankUser", "EmbeddingsUser"):
        assert getattr(module, name).wait_time(None) == 0.5


def test_a_split_topology_without_endpoint_urls_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The RAG system would refuse this configuration; the load generator never
    # builds it, so Locust can still drive whatever answers at --host.
    monkeypatch.setenv("RAG_TOPOLOGY", "split-all")
    monkeypatch.delenv("RAG_EMBEDDING_API_URL", raising=False)
    monkeypatch.delenv("RAG_RERANK_API_URL", raising=False)

    module = _import_locustfile("locustfile_split_topology")

    assert module.SETTINGS.target == "rag"
    assert not module.RagUser.abstract


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
    assert json.loads((tmp_path / "monolith_target.json").read_text()) == {
        "target": "rag",
        "seen": [{"topology": "monolith", "run_label": "mono", "responses": 2}],
    }
    locustfile.write_target_record_on_stop(SimpleNamespace(parsed_options=None))


# --- the stats file holds the finished run ----------------------------------------


def locust_writes_its_stats_file(prefix: Path, stats: RequestStats) -> None:
    """What Locust's own writer leaves on disk at one of its once-a-second writes."""
    environment = SimpleNamespace(stats=stats)
    with Path(f"{prefix}_stats.csv").open("w") as handle:
        StatsCSV(environment, PERCENTILES_TO_REPORT).requests_csv(csv.writer(handle))


def test_stop_listener_rewrites_the_stats_file_with_the_last_second(
    locustfile: ModuleType, tmp_path: Path
) -> None:
    prefix, stats = tmp_path / "monolith", RequestStats()
    stats.log_request("POST", "/query", 10, 100)
    stats.log_request("POST", "/query", 20, 100)
    locust_writes_its_stats_file(prefix, stats)  # its last write before the run ends
    stats.log_request("POST", "/query", 30, 100)  # answered after that write
    stats.log_request("POST", "/query", 40, 100)
    stats.log_error("POST", "/query", "HTTP 503: down")
    assert load_stats(prefix)["POST /query"].requests == 2  # what rag-compare saw

    options = SimpleNamespace(csv_prefix=str(prefix))
    environment = SimpleNamespace(parsed_options=options, stats=stats)
    locustfile.write_final_stats_on_stop(environment)

    final = load_stats(prefix)
    assert (final["POST /query"].requests, final["POST /query"].failures) == (4, 1)
    assert final["Aggregated"].requests == 4
    assert final["POST /query"].p50 == stats.get("/query", "POST").median_response_time


def test_the_rewritten_stats_file_has_the_columns_locust_writes(
    locustfile: ModuleType, tmp_path: Path
) -> None:
    prefix, stats = tmp_path / "monolith", RequestStats()
    stats.log_request("POST", "/retrieve", 12, 100)
    locust_writes_its_stats_file(prefix, stats)
    written_by_locust = Path(f"{prefix}_stats.csv").read_text()

    options = SimpleNamespace(csv_prefix=str(prefix))
    locustfile.write_final_stats_on_stop(
        SimpleNamespace(parsed_options=options, stats=stats)
    )

    # Same statistics, so the same bytes: header, rows and line endings.
    assert Path(f"{prefix}_stats.csv").read_text() == written_by_locust


@pytest.mark.parametrize("options", [None, SimpleNamespace(csv_prefix=None)])
def test_no_stats_file_is_written_without_csv(
    locustfile: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: SimpleNamespace | None,
) -> None:
    monkeypatch.chdir(tmp_path)

    locustfile.write_final_stats_on_stop(SimpleNamespace(parsed_options=options))

    assert list(tmp_path.iterdir()) == []
