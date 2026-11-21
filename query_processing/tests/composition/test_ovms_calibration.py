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
from query_processing.composition.composition_root import CompositionRoot
from query_processing.composition.ovms_calibration import ovms_candidates
from query_processing.composition.ovms_stages import calibration_http_policy
from query_processing.config.apps.ovms_settings import OvmsSettings
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.infrastructure.http.http_policy import HttpPolicy
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
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
def test_the_candidates_on_one_thread_a_core(
    cpus: int,
    streams: int,
    threads: int,
    expected: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    found = ovms_candidates(
        cpus, streams=streams, threads=threads, cores=cpus, hinted=False
    )

    assert found == expected


def test_stream_candidates_stop_at_the_cores() -> None:
    # Measured: 16 streams on 8 cores took 14.8 s where 1 to 4 took 3.8 s.
    assert ovms_candidates(16, streams=0, threads=0, cores=8, hinted=False) == (
        (1, 2, 4, 8),
        (16, 8, 4, 2, 1),
    )
    assert ovms_candidates(2, streams=0, threads=0, cores=1, hinted=False) == (
        (1,),
        (2, 1),
    )
    assert ovms_candidates(12, streams=0, threads=0, cores=6, hinted=False) == (
        (1, 3, 6),
        (12, 6, 3, 1),
    )


def test_a_pinned_stream_count_above_the_cores_is_still_tried() -> None:
    # A pin is the owner's choice.
    streams, _ = ovms_candidates(4, streams=4, threads=0, cores=2, hinted=False)

    assert streams == (4,)


def test_under_a_performance_hint_streams_are_not_searched() -> None:
    # OVMS refuses NUM_STREAMS beside a hint, so the hint sets the streams.
    assert ovms_candidates(8, streams=0, threads=0, cores=4, hinted=True) == (
        (1,),
        (8, 4, 2, 1),
    )


# --- against the stand-in ------------------------------------------------------------


@pytest.fixture
def models(tmp_path: Path, settings_factory: SettingsFactory) -> Path:
    settings = settings_factory()
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", settings.components["reranker"].alias)
    write_ovms_export(models, "embedder", settings.components["embedder"].alias)
    return models


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in ``ovms`` first on PATH; its starts are recorded here."""
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record


def two_cpus(
    settings_factory: SettingsFactory, **overrides: object
) -> WorkflowSettings:
    """2 CPUs of one thread each, so 2 cores: two stream counts to try."""
    return settings_factory(
        local_models__cpus=2,
        local_models__threads_per_core=1,
        **FAKE_BACKENDS,
        **overrides,
    )


def ovms_settings(**pinned: object) -> OvmsSettings:
    """The OVMS App's own settings, not read from a stray ``.env``."""
    return OvmsSettings(_env_file=None, start_timeout_s=20.0, **pinned)  # type: ignore[arg-type]


def graph_plugin_config(config: Path) -> dict[str, str]:
    model = json.loads(config.read_text())["model_config_list"][0]["config"]
    graph = (config.parent / model["base_path"] / "graph.pbtxt").read_text()
    [found] = re.findall(r"plugin_config: '([^']*)'", graph)
    return dict(json.loads(found))


def logged(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
    lines = [json.loads(record.message) for record in caplog.records]
    return [line for line in lines if line["event"] == event]


async def serving_config(
    settings: WorkflowSettings,
    ovms: OvmsSettings,
    component: str,
    models: Path,
    serving: Path,
) -> Path:
    async with asyncio.timeout(BOUND_S):
        async with CompositionRoot(settings) as composition:
            return await composition.ovms_serving_config(
                component, ovms, models, serving
            )


@pytest.mark.parametrize("component", ["reranker", "embedder"])
async def test_start_up_calibrates_then_writes_the_folder_for_the_choice(
    settings_factory: SettingsFactory,
    models: Path,
    record: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    component: str,
) -> None:
    serving = tmp_path / "serving"
    settings = two_cpus(settings_factory)
    with caplog.at_level("INFO", logger="query_processing"):
        config = await serving_config(
            settings, ovms_settings(), component, models, serving
        )

    assert config == serving / f"config_{component}.json"
    starts = ovms_starts(record)
    assert [start["plugin_config"] for start in starts[:2]] == ALL_ON_2_CPUS
    [line] = logged(caplog, "ovms_serving_config")
    assert line["calibrated"] is True
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
        "weight-format",
        settings.components["reranker"].alias,
        settings.components["embedder"].alias,
    }  # nothing written into the export


async def test_both_pinned_starts_nothing(
    settings_factory: SettingsFactory,
    models: Path,
    record: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings, ovms = two_cpus(settings_factory), ovms_settings(num_streams=2, threads=2)
    with caplog.at_level("INFO", logger="query_processing"):
        config = await serving_config(
            settings, ovms, "reranker", models, tmp_path / "s"
        )

    assert ovms_starts(record) == []
    assert graph_plugin_config(config) == ALL_ON_2_CPUS[1]
    [line] = logged(caplog, "ovms_serving_config")
    keys = ("component", "streams", "threads", "calibrated")
    assert {k: line[k] for k in keys} == {
        "component": "reranker",
        "streams": 2,
        "threads": 2,
        "calibrated": False,
    }
    assert line["config"] == str(config)


async def test_one_pinned_parameter_searches_only_the_other(
    settings_factory: SettingsFactory, models: Path, record: Path, tmp_path: Path
) -> None:
    settings, ovms = two_cpus(settings_factory), ovms_settings(threads=2)
    await serving_config(settings, ovms, "embedder", models, tmp_path / "s")

    configs = [start["plugin_config"] for start in ovms_starts(record)]
    assert configs == ALL_ON_2_CPUS  # the streams on the pinned threads, no more


async def test_on_demand_searches_both_even_when_pinned_and_cleans_up(
    settings_factory: SettingsFactory, models: Path, record: Path
) -> None:
    settings, ovms = two_cpus(settings_factory), ovms_settings(num_streams=1, threads=1)
    async with asyncio.timeout(BOUND_S):
        async with CompositionRoot(settings) as composition:
            calibration = await composition.ovms_calibration("reranker", ovms, models)

    starts = ovms_starts(record)
    assert [start["plugin_config"] for start in starts[:2]] == ALL_ON_2_CPUS
    chosen = calibration.report.chosen
    assert calibration.pinned == {
        "QP_OVMS_NUM_STREAMS": chosen.streams,
        "QP_OVMS_THREADS": chosen.threads,
    }
    assert (calibration.component, calibration.runtime) == ("reranker", "ovms")
    assert calibration.cpus == 2.0
    assert (calibration.report.stage, calibration.report.requests) == ("rerank", 4)
    reranker = settings.components["reranker"]
    assert chosen.batch_size == reranker.local.calibration_records
    served_from = Path(str(starts[0]["argv"][-1]))  # type: ignore[index]
    assert not served_from.exists()  # the temporary serving folder is gone


# --- the cores, the performance hint, the graph's options --------------------------


async def test_two_cpus_of_one_core_try_one_stream(
    settings_factory: SettingsFactory, models: Path, record: Path, tmp_path: Path
) -> None:
    # threads_per_core is 2 by default: 2 CPUs are one core, so one stream.
    settings = settings_factory(local_models__cpus=2, **FAKE_BACKENDS)

    await serving_config(settings, ovms_settings(), "reranker", models, tmp_path / "s")

    streams = {start["plugin_config"]["NUM_STREAMS"] for start in ovms_starts(record)}  # type: ignore[index]
    assert streams == {"1"}


async def test_a_performance_hint_searches_threads_and_writes_no_stream_count(
    settings_factory: SettingsFactory,
    models: Path,
    record: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ovms = ovms_settings(openvino={"performance_hint": "THROUGHPUT"})
    with caplog.at_level("INFO", logger="query_processing"):
        config = await serving_config(
            two_cpus(settings_factory), ovms, "embedder", models, tmp_path / "s"
        )

    configs = [start["plugin_config"] for start in ovms_starts(record)]
    assert {c["INFERENCE_NUM_THREADS"] for c in configs} == {"2", "1"}  # type: ignore[index]
    assert all("NUM_STREAMS" not in c for c in configs)  # type: ignore[operator]
    written = graph_plugin_config(config)
    assert "NUM_STREAMS" not in written
    assert written["PERFORMANCE_HINT"] == "THROUGHPUT"
    [line] = logged(caplog, "ovms_serving_config")
    assert line["streams"] is None  # the hint's, not a count of ours


async def test_on_demand_under_a_hint_pins_no_stream_count(
    settings_factory: SettingsFactory, models: Path, record: Path
) -> None:
    ovms = ovms_settings(openvino={"performance_hint": "THROUGHPUT"})
    async with asyncio.timeout(BOUND_S):
        async with CompositionRoot(two_cpus(settings_factory)) as composition:
            calibration = await composition.ovms_calibration("reranker", ovms, models)

    assert list(calibration.pinned) == ["QP_OVMS_THREADS"]


async def test_the_graph_options_reach_the_served_graph(
    settings_factory: SettingsFactory, models: Path, record: Path, tmp_path: Path
) -> None:
    ovms = ovms_settings(
        num_streams=1,
        threads=1,
        graph_queue_max_size="2",
        embeddings_truncate=True,
        embeddings_max_length=128,
    )

    config = await serving_config(
        two_cpus(settings_factory), ovms, "embedder", models, tmp_path / "s"
    )

    model = json.loads(config.read_text())["model_config_list"][0]["config"]
    graph = (config.parent / model["base_path"] / "graph.pbtxt").read_text()
    assert graph.startswith("# OVMS_GRAPH_QUEUE_MAX_SIZE: 2\n")
    assert "truncate: true," in graph
    assert "max_length: 128," in graph


# --- a slow candidate --------------------------------------------------------------


def test_the_calibration_requests_wait_up_to_the_budget_with_no_retry(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(http__timeout_s=30.0, calibration__budget_s=600.0)

    policy = calibration_http_policy(settings.calibration)

    assert policy == HttpPolicy(timeout_s=600.0, retries=0, backoff_s=0.0)


async def test_a_candidate_slower_than_the_budget_is_kept_untimed_not_a_failure(
    settings_factory: SettingsFactory,
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # GCP sitting of 2026-11-15: 16 requests on one stream, about 2 s each, so
    # the last waited 32 s and the 30 s HTTP timeout failed the App. Here 4
    # requests of 0.5 s on one stream take 2 s: past the HTTP timeout and the
    # budget, while the readiness request (one, 0.5 s) is within both.
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record, request_s=0.5)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    settings = two_cpus(
        settings_factory, http__timeout_s=1.0, calibration__budget_s=1.5
    )
    with caplog.at_level("INFO", logger="query_processing"):
        config = await serving_config(
            settings, ovms_settings(), "reranker", models, tmp_path / "s"
        )

    assert len(ovms_starts(record)) == 1  # the search ended with the first
    assert graph_plugin_config(config) == ALL_ON_2_CPUS[0]
    [calibrated] = logged(caplog, "stage_calibrated")
    assert (calibrated["trials"], calibrated["cut_short"]) == ([], True)
    [line] = logged(caplog, "ovms_serving_config")
    assert (line["streams"], line["threads"], line["calibrated"]) == (1, 2, False)


# --- the weight format of the export -----------------------------------------------


async def test_an_export_of_another_weight_format_is_refused_before_ovms_starts(
    settings_factory: SettingsFactory, models: Path, record: Path, tmp_path: Path
) -> None:
    (models / "weight-format").write_text("int8\n")
    settings = two_cpus(settings_factory)  # asks fp32, the default

    with pytest.raises(
        ValueError, match="weight format int8, but fp32 was asked"
    ) as refusal:
        await serving_config(
            settings, ovms_settings(), "reranker", models, tmp_path / "s"
        )

    assert ovms_starts(record) == []
    message = str(refusal.value)
    assert str(models) in message
    # Both ways out: serve what the folder holds, or export into another folder.
    assert "QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT=int8" in message
    assert "QP_OVMS_MODELS_DIR" in message


async def test_an_export_without_a_record_is_served_when_int8_is_asked(
    settings_factory: SettingsFactory, models: Path, record: Path, tmp_path: Path
) -> None:
    (models / "weight-format").unlink()  # exported before the record: int8
    settings = two_cpus(settings_factory, local_models__openvino_weight_format="int8")
    pinned = ovms_settings(num_streams=1, threads=1)

    config = await serving_config(settings, pinned, "reranker", models, tmp_path / "s")

    assert config == tmp_path / "s" / "config_reranker.json"
