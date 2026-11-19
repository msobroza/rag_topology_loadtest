"""``OpenVinoCpuSettings``: OpenVINO's CPU properties, named or passed through.

One block read from two homes: the workflow file's ``local_models.openvino``
(``QP_LOCAL_MODELS__OPENVINO__*``), for the in-process models, and an OVMS
App's ``QP_OVMS_OPENVINO__*``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from query_processing.config.apps.ovms_settings import OvmsSettings
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.openvino_cpu_settings import OpenVinoCpuSettings
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.infrastructure.openvino_cpu_properties import (
    openvino_cpu_properties,
)
from rag_components import shell_settings
from rag_graphs import fixture_graph, write_workflow_file

SettingsFactory = Callable[..., WorkflowSettings]


@pytest.fixture(autouse=True)
def _no_qp_variable_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in shell_settings(os.environ):
        monkeypatch.delenv(name)


def test_the_defaults_write_what_runs_today() -> None:
    assert openvino_cpu_properties(4, OpenVinoCpuSettings().chosen()) == {
        "INFERENCE_NUM_THREADS": "4",
        "ENABLE_HYPER_THREADING": "YES",
    }


@pytest.mark.parametrize(
    ("named", "passed"),
    [
        ({"performance_hint": "THROUGHPUT"}, {"PERFORMANCE_HINT": "THROUGHPUT"}),
        ({"cpu_pinning": False}, {"ENABLE_CPU_PINNING": "NO"}),
        ({"cpu_reservation": True}, {"ENABLE_CPU_RESERVATION": "YES"}),
        ({"inference_precision_hint": "f32"}, {"INFERENCE_PRECISION_HINT": "f32"}),
        ({"execution_mode_hint": "ACCURACY"}, {"EXECUTION_MODE_HINT": "ACCURACY"}),
        (
            {"performance_hint": "THROUGHPUT", "performance_hint_num_requests": 2},
            {"PERFORMANCE_HINT": "THROUGHPUT", "PERFORMANCE_HINT_NUM_REQUESTS": "2"},
        ),
    ],
)
def test_a_property_named_or_passed_through_gives_the_same_properties(
    named: dict[str, object], passed: dict[str, str]
) -> None:
    by_name = OpenVinoCpuSettings.model_validate(named).chosen()

    assert by_name == OpenVinoCpuSettings(properties=passed).chosen()
    assert by_name == {"ENABLE_HYPER_THREADING": "YES", **passed}


def test_the_named_properties_come_first_then_the_pass_through() -> None:
    chosen = OpenVinoCpuSettings(
        cpu_pinning=True, properties={"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}
    ).chosen()

    assert list(chosen) == [
        "ENABLE_HYPER_THREADING",
        "ENABLE_CPU_PINNING",
        "SCHEDULING_CORE_TYPE",
    ]


def test_hyper_threading_passed_through_replaces_the_default() -> None:
    chosen = OpenVinoCpuSettings(properties={"ENABLE_HYPER_THREADING": "NO"}).chosen()

    assert chosen == {"ENABLE_HYPER_THREADING": "NO"}


def test_a_property_with_no_named_setting_is_written_as_it_is() -> None:
    chosen = OpenVinoCpuSettings(
        properties={"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}
    ).chosen()

    assert chosen["SCHEDULING_CORE_TYPE"] == "PCORE_ONLY"


def test_a_property_given_both_ways_is_refused_naming_both() -> None:
    with pytest.raises(ValueError, match=r"hyper_threading.*ENABLE_HYPER_THREADING"):
        OpenVinoCpuSettings(
            hyper_threading=True, properties={"ENABLE_HYPER_THREADING": "NO"}
        )


@pytest.mark.parametrize("calibrated", ["NUM_STREAMS", "INFERENCE_NUM_THREADS"])
def test_a_property_calibration_owns_is_refused_in_the_pass_through(
    calibrated: str,
) -> None:
    with pytest.raises(ValueError, match=calibrated):
        OpenVinoCpuSettings(properties={calibrated: "1"})


@pytest.mark.parametrize(
    "values",
    [
        {"performance_hint_num_requests": 2},
        {"performance_hint": "LATENCY", "performance_hint_num_requests": 2},
        {"properties": {"PERFORMANCE_HINT_NUM_REQUESTS": "2"}},
    ],
)
def test_a_request_count_without_the_throughput_hint_is_refused(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="THROUGHPUT"):
        OpenVinoCpuSettings.model_validate(values)


def test_a_request_count_under_a_hint_passed_through_is_taken() -> None:
    chosen = OpenVinoCpuSettings(
        performance_hint_num_requests=3, properties={"PERFORMANCE_HINT": "THROUGHPUT"}
    ).chosen()

    assert chosen["PERFORMANCE_HINT_NUM_REQUESTS"] == "3"


@pytest.mark.parametrize(
    "values",
    [
        {"performance_hint": "FAST"},
        {"inference_precision_hint": "f16"},
        {"execution_mode_hint": "SPEED"},
        {"performance_hint_num_requests": -1},
        {"hyper_thread": False},
    ],
)
def test_a_value_or_key_the_block_does_not_have_is_refused(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        OpenVinoCpuSettings.model_validate(values)


# --- the two homes ------------------------------------------------------------------


def read_local_models_block(settings_factory: SettingsFactory) -> OpenVinoCpuSettings:
    return settings_factory().local_models.openvino


def read_ovms_block(_: SettingsFactory) -> OpenVinoCpuSettings:
    return OvmsSettings(_env_file=None).openvino  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("variable", "read"),
    [
        ("QP_LOCAL_MODELS__OPENVINO__HYPER_THREADING", read_local_models_block),
        ("QP_OVMS_OPENVINO__HYPER_THREADING", read_ovms_block),
    ],
)
def test_the_block_is_read_from_both_homes(
    variable: str,
    read: Callable[[SettingsFactory], OpenVinoCpuSettings],
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(variable, "false")

    assert read(settings_factory).chosen() == {"ENABLE_HYPER_THREADING": "NO"}


@pytest.mark.parametrize(
    ("variable", "read"),
    [
        ("QP_LOCAL_MODELS__OPENVINO__PROPERTIES", read_local_models_block),
        ("QP_OVMS_OPENVINO__PROPERTIES", read_ovms_block),
    ],
)
def test_the_pass_through_is_read_as_a_json_object_from_both_homes(
    variable: str,
    read: Callable[[SettingsFactory], OpenVinoCpuSettings],
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(variable, '{"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}')

    assert read(settings_factory).chosen()["SCHEDULING_CORE_TYPE"] == "PCORE_ONLY"


def test_the_block_is_read_from_the_workflow_file(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = write_workflow_file(tmp_path / "workflow.yaml", fixture_graph())
    with workflow.open("a", encoding="utf-8") as file:
        file.write("local_models: {openvino: {cpu_pinning: false}, workers: 2}\n")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    local_models = settings_factory().local_models

    assert local_models.workers == 2
    assert local_models.openvino.chosen() == {
        "ENABLE_HYPER_THREADING": "YES",
        "ENABLE_CPU_PINNING": "NO",
    }


def test_a_key_the_block_does_not_have_is_refused_by_its_variable(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_LOCAL_MODELS__OPENVINO__HYPER_THREAD", "false")

    with pytest.raises(ValidationError, match="hyper_thread"):
        settings_factory()


ENV_FILE = """\
QP_OVMS_MODELS_DIR=/mnt/data/ovms-models
QP_OVMS_BUNDLE=/mnt/data/ovms.tar.gz
QP_OVMS_PREFIX_PROXY=1
QP_OVMS_NUM_STREAMS=2
QP_OVMS_OPENVINO__HYPER_THREADING=false
QP_OVMS_OPENVINO__PROPERTIES={"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}
QP_LOCAL_MODELS__OPENVINO__CPU_PINNING=true
QP_LOCAL_MODELS__WORKERS=2
QP_COMPONENTS__RERANKER__BACKEND=fake
QP_HTTP__RETRIES=0
HF_HUB_OFFLINE=1
"""


def test_an_ovms_app_reads_its_block_from_a_dotenv_holding_other_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(ENV_FILE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    ovms = OvmsSettings()

    assert ovms.num_streams == 2
    assert ovms.openvino.chosen() == {
        "ENABLE_HYPER_THREADING": "NO",
        "SCHEDULING_CORE_TYPE": "PCORE_ONLY",
    }


def test_the_workflow_app_reads_its_own_block_from_the_same_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(ENV_FILE, encoding="utf-8")

    local_models = WorkflowSettings(_env_file=env_file).local_models  # type: ignore[call-arg]

    assert local_models.workers == 2
    assert local_models.openvino.chosen() == {
        "ENABLE_HYPER_THREADING": "YES",
        "ENABLE_CPU_PINNING": "YES",
    }
