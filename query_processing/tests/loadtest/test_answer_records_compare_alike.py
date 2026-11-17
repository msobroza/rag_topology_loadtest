"""``qp-record-answers`` then ``qp-compare-answers``: what answers the same compares so.

Each test records the workflow app in process, on the fixture workflow file
with every component on ``fake``, and compares the records.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from answer_record_helpers import (
    compare,
    record,
    use_loadtest_variables,
    workflow_client,
)
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_model_server import create_model_server
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
RERANK_URL = "http://reranker-app.test/v3/rerank"


@pytest.fixture(autouse=True)
def _loadtest_variables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_loadtest_variables(monkeypatch, tmp_path, {})


def test_the_same_app_recorded_twice_compares_as_identical(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "before") == 0
        assert record(client, tmp_path / "after") == 0

    status = compare(tmp_path / "before", tmp_path / "after", out=tmp_path / "c.md")

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 0
    assert "| after | 6 | 6 | 0 |" in report


def test_the_app_under_two_topology_aliases_compares_as_identical(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    for alias in ("monolith", "split-all"):
        settings = settings_factory(**FAKE_BACKENDS, topology_alias=alias)
        with workflow_client(settings) as client:
            assert record(client, tmp_path / alias) == 0

    status = compare(
        tmp_path / "monolith", tmp_path / "split-all", out=tmp_path / "c.md"
    )

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 0
    assert "topology `split-all`" in report
    assert "| split-all | 6 | 6 | 0 |" in report


def test_a_reranker_served_by_the_model_server_answers_as_the_one_in_process(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "in-process") == 0
    model_server = create_model_server(
        "reranker",
        CompositionRoot(settings_factory(**FAKE_BACKENDS), reranker=FakeReranker()),
    )
    served = settings_factory(
        **{
            **FAKE_BACKENDS,
            "components__reranker__backend": "served",
            "components__reranker__served__url": RERANK_URL,
        }
    )
    with TestClient(model_server):  # runs its lifespan
        transport = httpx.ASGITransport(app=model_server)
        with workflow_client(served, transport) as client:
            assert record(client, tmp_path / "served") == 0

    status = compare(
        tmp_path / "in-process", tmp_path / "served", out=tmp_path / "c.md"
    )

    report = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert status == 0, report
    assert "reranker=served" in report  # the run label: the call went over HTTP
    assert "| served | 6 | 6 | 0 |" in report
