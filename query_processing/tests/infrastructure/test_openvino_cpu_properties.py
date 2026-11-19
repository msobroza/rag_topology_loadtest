"""The OpenVINO CPU properties: at the defaults, byte for byte what ran before the block."""

from __future__ import annotations

import json

import pytest

from query_processing.config.openvino_cpu_settings import OpenVinoCpuSettings
from query_processing.domain.runtime.calibration import StageTuning
from query_processing.infrastructure.local_models.local_model import LocalModel
from query_processing.infrastructure.openvino_cpu_properties import (
    openvino_cpu_properties,
)
from query_processing.infrastructure.ovms.ovms_graph import ovms_plugin_config

# What the block's defaults choose: the properties every App ran before it.
DEFAULTS = OpenVinoCpuSettings().chosen()


def test_an_in_process_models_ov_config_at_the_defaults_is_todays() -> None:
    model = LocalModel("m", "m", "openvino", DEFAULTS)
    ov_config = model.load_arguments(4)["model_kwargs"]["ov_config"]

    assert json.dumps(ov_config) == (
        '{"INFERENCE_NUM_THREADS": "4", "ENABLE_HYPER_THREADING": "YES"}'
    )


def test_ovms_plugin_config_at_the_defaults_is_todays() -> None:
    config = ovms_plugin_config(
        StageTuning(threads=4, batch_size=20, streams=2), DEFAULTS
    )

    assert json.dumps(config) == (
        '{"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "4", '
        '"ENABLE_HYPER_THREADING": "YES"}'
    )


def test_the_blocks_properties_follow_the_thread_count() -> None:
    properties = openvino_cpu_properties(2, {"PERFORMANCE_HINT": "LATENCY"})

    assert properties == {"INFERENCE_NUM_THREADS": "2", "PERFORMANCE_HINT": "LATENCY"}


def test_fewer_than_one_thread_is_refused() -> None:
    with pytest.raises(ValueError, match="threads=0"):
        openvino_cpu_properties(0, DEFAULTS)
