"""OVMS calibration through ``CompositionRoot``, against a stand-in ``ovms`` on PATH.

The stand-in holds ``NUM_STREAMS`` request slots, so with 4 requests in flight
2 streams answer sooner than 1; which candidate wins is still the real clock's
call, so the tests assert the candidates tried and that the folder written
holds whatever was chosen.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from ovms_stand_ins import ovms_starts, write_ovms_export, write_ovms_stand_in
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.composition.ovms_calibration import ovms_candidates
from rag_load_test.composition.ovms_stages import calibration_http_policy
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.roles import ModelServerPart
from rag_load_test.infrastructure.http.http_policy import HttpPolicy

SettingsFactory = Callable[..., RagSettings]
BOUND_S = 60.0  # no test here may take longer
ALL_ON_2_CPUS = [
    {"NUM_STREAMS": "1", "INFERENCE_NUM_THREADS": "2", "ENABLE_HYPER_THREADING": "YES"},
    {"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "2", "ENABLE_HYPER_THREADING": "YES"},
]


# --- the candidates ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cpus", "streams", "threads", "expected"),
    [
        (4, 0, 0, ((1, 2, 4), (4, 2, 1))),
        (6, 0, 0, ((1, 3, 6), (6, 3, 1))),
        (1, 0, 0, ((1,), (1,))),
        (8, 0, 2, ((1, 2), (2,))),  # no stream count above the pinned threads
        (8, 2, 0, ((2,), (8, 4, 2))),  # no thread count below the pinned streams
        (2, 4, 0, ((4,), (4,))),  # pinned above the CPUs: one thread per stream
        (8, 2, 4, ((2,), (4,))),
    ],
)
def test_the_candidates(
    cpus: int,
    streams: int,
    threads: int,
    expected: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    assert ovms_candidates(cpus, streams=streams, threads=threads) == expected


# --- against the stand-in ------------------------------------------------------------


@pytest.fixture
def models(tmp_path: Path, settings_factory: SettingsFactory) -> Path:
    settings = settings_factory()
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", settings.rerank_api_model)
    write_ovms_export(models, "embedder", settings.embedding_api_model)
    return models


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in ``ovms`` first on PATH; its starts are recorded here."""
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record


def two_cpus(settings_factory: SettingsFactory, **overrides: object) -> RagSettings:
    return settings_factory(
        model_cpus=2,
        embedding_backend="fake",
        rerank_backend="fake",
        vector_store_backend="fake",
        generation_backend="fake",
        ovms_start_timeout_s=20.0,
        **overrides,
    )


def graph_plugin_config(config: Path) -> dict[str, str]:
    model = json.loads(config.read_text())["model_config_list"][0]["config"]
    graph = (config.parent / model["base_path"] / "graph.pbtxt").read_text()
    [found] = re.findall(r"plugin_config: '([^']*)'", graph)
    return dict(json.loads(found))


def graph_normalize(config: Path) -> list[str]:
    model = json.loads(config.read_text())["model_config_list"][0]["config"]
    graph = (config.parent / model["base_path"] / "graph.pbtxt").read_text()
    return re.findall(r"normalize_embeddings: (\w+)", graph)


def logged(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
    lines = [json.loads(record.message) for record in caplog.records]
    return [line for line in lines if line["event"] == event]


async def serving_config(
    settings: RagSettings, part: ModelServerPart, models: Path, serving: Path
) -> Path:
    async with asyncio.timeout(BOUND_S):
        async with CompositionRoot(settings) as composition:
            return await composition.ovms_serving_config(part, models, serving)


@pytest.mark.parametrize("part", ["reranker", "embedder"])
async def test_start_up_calibrates_then_writes_the_folder_for_the_choice(
    settings_factory: SettingsFactory,
    models: Path,
    record: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    part: ModelServerPart,
) -> None:
    serving = tmp_path / "serving"
    with caplog.at_level("INFO", logger="rag_load_test"):
        config = await serving_config(two_cpus(settings_factory), part, models, serving)

    assert config == serving / f"config_{part}.json"
    starts = ovms_starts(record)
    assert [start["plugin_config"] for start in starts[:2]] == ALL_ON_2_CPUS
    # RAG_EMBEDDING_NORMALIZE unset: every embedder candidate served unnormalised.
    served = ["false"] if part == "embedder" else []
    assert all(start["normalize"] == served for start in starts)
    assert graph_normalize(config) == served
    [line] = logged(caplog, "ovms_serving_config")
    assert line["calibrated"] is True
    assert line.get("normalize_embeddings") == (False if served else None)
    assert graph_plugin_config(config) == {
        "NUM_STREAMS": str(line["streams"]),
        "INFERENCE_NUM_THREADS": str(line["threads"]),
        "ENABLE_HYPER_THREADING": "YES",
    }
    [calibrated] = logged(caplog, "stage_calibrated")
    assert calibrated["requests"] == 4
    assert {p.name for p in models.iterdir()} == {
        "config_reranker.json",
        "config_embedder.json",
        two_cpus(settings_factory).rerank_api_model,
        two_cpus(settings_factory).embedding_api_model,
    }  # nothing written into the export


async def test_both_pinned_starts_nothing(
    settings_factory: SettingsFactory,
    models: Path,
    record: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = two_cpus(settings_factory, ovms_num_streams=2, ovms_threads=2)
    with caplog.at_level("INFO", logger="rag_load_test"):
        config = await serving_config(settings, "reranker", models, tmp_path / "s")

    assert ovms_starts(record) == []
    assert graph_plugin_config(config) == ALL_ON_2_CPUS[1]
    [line] = logged(caplog, "ovms_serving_config")
    assert {k: line[k] for k in ("part", "streams", "threads", "calibrated")} == {
        "part": "reranker",
        "streams": 2,
        "threads": 2,
        "calibrated": False,
    }
    assert line["config"] == str(config)


@pytest.mark.parametrize("normalize", [False, True])
async def test_the_embedder_is_served_with_the_normalize_setting(
    settings_factory: SettingsFactory,
    models: Path,
    record: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    normalize: bool,
) -> None:
    settings = two_cpus(settings_factory, ovms_threads=2, embedding_normalize=normalize)
    with caplog.at_level("INFO", logger="rag_load_test"):
        config = await serving_config(settings, "embedder", models, tmp_path / "s")

    expected = ["true" if normalize else "false"]
    assert [start["normalize"] for start in ovms_starts(record)] == [expected] * 2
    assert graph_normalize(config) == expected
    [line] = logged(caplog, "ovms_serving_config")
    assert line["normalize_embeddings"] is normalize


async def test_the_reranker_graph_and_event_carry_no_normalization(
    settings_factory: SettingsFactory,
    models: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = two_cpus(
        settings_factory, ovms_num_streams=2, ovms_threads=2, embedding_normalize=True
    )
    with caplog.at_level("INFO", logger="rag_load_test"):
        config = await serving_config(settings, "reranker", models, tmp_path / "s")

    exported = (models / settings.rerank_api_model / "graph.pbtxt").read_text()
    served = (config.parent / settings.rerank_api_model / "graph.pbtxt").read_text()
    assert served.replace(json.dumps(ALL_ON_2_CPUS[1]), '{"NUM_STREAMS": "1" }') == (
        exported
    )
    [line] = logged(caplog, "ovms_serving_config")
    assert "normalize_embeddings" not in line


async def test_one_pinned_parameter_searches_only_the_other(
    settings_factory: SettingsFactory, models: Path, record: Path, tmp_path: Path
) -> None:
    settings = two_cpus(settings_factory, ovms_threads=2)
    await serving_config(settings, "embedder", models, tmp_path / "s")

    configs = [start["plugin_config"] for start in ovms_starts(record)]
    assert configs == ALL_ON_2_CPUS  # the streams on the pinned threads, no more


async def test_on_demand_searches_both_even_when_pinned_and_cleans_up(
    settings_factory: SettingsFactory, models: Path, record: Path
) -> None:
    settings = two_cpus(settings_factory, ovms_num_streams=1, ovms_threads=1)
    async with asyncio.timeout(BOUND_S):
        async with CompositionRoot(settings) as composition:
            calibration = await composition.ovms_calibration("reranker", models)

    starts = ovms_starts(record)
    assert [start["plugin_config"] for start in starts[:2]] == ALL_ON_2_CPUS
    chosen = calibration.report.chosen
    assert calibration.pinned == {
        "RAG_OVMS_NUM_STREAMS": chosen.streams,
        "RAG_OVMS_THREADS": chosen.threads,
    }
    assert (calibration.runtime, calibration.cpus) == ("ovms", 2.0)
    assert (calibration.report.stage, calibration.report.requests) == ("rerank", 4)
    assert chosen.batch_size == settings.top_k_retrieve
    served_from = Path(str(starts[0]["argv"][-1]))  # type: ignore[index]
    assert not served_from.exists()  # the temporary serving folder is gone


# --- a slow candidate --------------------------------------------------------------


def test_the_calibration_requests_wait_up_to_the_budget_with_no_retry(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(http_timeout_s=30.0, calibration_budget_s=600.0)

    policy = calibration_http_policy(settings)

    assert policy == HttpPolicy(timeout_s=600.0, retries=0, backoff_s=0.0)


async def test_a_candidate_slower_than_the_budget_is_kept_untimed_not_a_failure(
    settings_factory: SettingsFactory,
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # GCP sitting of 2026-11-15: 16 requests on one stream, about 2 s each, so
    # the last waited 32 s and RAG_HTTP_TIMEOUT_S=30 failed the App. Here 4
    # requests of 0.5 s on one stream take 2 s: past the HTTP timeout and the
    # budget, while the readiness request (one, 0.5 s) is within both.
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record, request_s=0.5)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    settings = two_cpus(settings_factory, http_timeout_s=1.0, calibration_budget_s=1.5)
    with caplog.at_level("INFO", logger="rag_load_test"):
        config = await serving_config(settings, "reranker", models, tmp_path / "s")

    assert len(ovms_starts(record)) == 1  # the search ended with the first
    assert graph_plugin_config(config) == ALL_ON_2_CPUS[0]
    [calibrated] = logged(caplog, "stage_calibrated")
    assert (calibrated["trials"], calibrated["cut_short"]) == ([], True)
    [line] = logged(caplog, "ovms_serving_config")
    assert (line["streams"], line["threads"], line["calibrated"]) == (1, 2, False)
