"""What the components share: four top-level blocks of the workflow file.

Each block is read from the workflow file and from its environment path, a
variable wins over the file key by key, and a key a block does not have is
refused. The flat variables of before are read no more.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.openvino_cpu_settings import OpenVinoCpuSettings
from query_processing.config.shared_settings import (
    CalibrationSettings,
    DefaultHttpSettings,
    DominoSettings,
    LocalModelsSettings,
)
from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import fixture_graph, write_workflow_file

SettingsFactory = Callable[..., WorkflowSettings]
SHARED_BLOCKS = """\
local_models: {runtime: torch, openvino_dir: /data/openvino, workers: 2, cpus: 3.5}
calibration: {budget_s: 12.5, rounds: 3}
http: {timeout_s: 9, retries: 1, backoff_s: 0.5}
domino: {token_url: "http://token.test/t", token_ttl_s: 60, tracing: true}
"""


@pytest.fixture
def shared_blocks_in_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A workflow file of the fixture's graph and components, with the four blocks."""
    workflow = write_workflow_file(tmp_path / "workflow.yaml", fixture_graph())
    with workflow.open("a", encoding="utf-8") as file:
        file.write(SHARED_BLOCKS)
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))


def test_the_shared_blocks_default_to_todays_values(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory()

    assert s.local_models == LocalModelsSettings(
        runtime="openvino",
        openvino_dir="models/openvino",
        openvino_weight_format="fp32",
        workers=4,
        cpus=0,
        threads_per_core=2,
        openvino=OpenVinoCpuSettings(),
    )
    assert s.calibration == CalibrationSettings(budget_s=90.0, rounds=1, users=4)
    assert s.http == DefaultHttpSettings(timeout_s=30.0, retries=2, backoff_s=0.2)
    assert s.domino == DominoSettings(
        token_url="http://localhost:8899/access-token", token_ttl_s=240.0, tracing=False
    )


@pytest.mark.usefixtures("shared_blocks_in_the_file")
def test_the_shared_blocks_are_read_from_the_workflow_file(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory()

    assert s.local_models == LocalModelsSettings(
        runtime="torch", openvino_dir="/data/openvino", workers=2, cpus=3.5
    )
    assert s.calibration == CalibrationSettings(budget_s=12.5, rounds=3)
    assert s.http == DefaultHttpSettings(timeout_s=9, retries=1, backoff_s=0.5)
    assert s.domino == DominoSettings(
        token_url="http://token.test/t", token_ttl_s=60, tracing=True
    )


@pytest.mark.usefixtures("shared_blocks_in_the_file")
def test_a_variable_wins_over_one_value_of_a_block_and_keeps_the_others(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_LOCAL_MODELS__CPUS", "8")
    monkeypatch.setenv("QP_HTTP__RETRIES", "0")
    monkeypatch.setenv("QP_DOMINO__TRACING", "false")
    monkeypatch.setenv("QP_CALIBRATION__ROUNDS", "5")

    s = settings_factory()

    assert (s.local_models.cpus, s.local_models.runtime) == (8, "torch")
    assert (s.http.retries, s.http.timeout_s) == (0, 9)
    assert (s.domino.tracing, s.domino.token_ttl_s) == (False, 60)
    assert (s.calibration.rounds, s.calibration.budget_s) == (5, 12.5)


def test_each_shared_value_names_the_variable_it_is_read_from() -> None:
    paths = {
        ("local_models", "runtime"): "QP_LOCAL_MODELS__RUNTIME",
        ("local_models", "openvino_dir"): "QP_LOCAL_MODELS__OPENVINO_DIR",
        ("calibration", "budget_s"): "QP_CALIBRATION__BUDGET_S",
        ("http", "backoff_s"): "QP_HTTP__BACKOFF_S",
        ("domino", "token_url"): "QP_DOMINO__TOKEN_URL",
    }

    assert {path: WorkflowSettings.variable_of(*path) for path in paths} == paths


@pytest.mark.parametrize(
    ("variable", "key"),
    [
        ("QP_LOCAL_MODELS__MODELS_DIR", "models_dir"),
        ("QP_HTTP__TIMEOUT", "timeout"),
        ("QP_DOMINO__TOKEN_TTL", "token_ttl"),
        ("QP_CALIBRATION__BUDGET", "budget"),
    ],
)
def test_a_key_a_shared_block_does_not_have_is_refused(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    key: str,
) -> None:
    monkeypatch.setenv(variable, "1")

    with pytest.raises(ValidationError, match=key):
        settings_factory()


def test_the_flat_variables_of_before_are_read_no_more(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "QP_VECTOR_STORE_BACKEND": "fake",
        "QP_ES_URL": "http://elsewhere:9200",
        "QP_CHROMA_PATH": "elsewhere",
        "QP_FAKE_STORE_SEED_DOCUMENTS": "3",
        "QP_LOCAL_MODEL_RUNTIME": "torch",
        "QP_OPENVINO_MODELS_DIR": "elsewhere",
        "QP_MODEL_WORKERS": "1",
        "QP_MODEL_CPUS": "1",
        "QP_CALIBRATION_ROUNDS": "9",
        "QP_HTTP_RETRIES": "0",
        "QP_DOMINO_TRACING": "true",
    }.items():
        monkeypatch.setenv(name, value)

    s = settings_factory()

    store = s.components["vector_store"]
    assert (store.backend, store.elasticsearch.url) == (
        "elasticsearch",
        "http://localhost:9200",
    )
    assert (store.chroma.path, store.fake.seed_documents) == ("chroma", 200)
    assert s.local_models == LocalModelsSettings()
    assert (s.calibration.rounds, s.http.retries, s.domino.tracing) == (1, 2, False)


# --- the OpenVINO weight format, the calibration users ---------------------------


def test_the_weight_format_defaults_to_fp32_and_reads_its_variable(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT", raising=False)
    assert settings_factory().local_models.openvino_weight_format == "fp32"

    monkeypatch.setenv("QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT", "int8")
    assert settings_factory().local_models.openvino_weight_format == "int8"


def test_a_weight_format_other_than_fp32_or_int8_is_rejected(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match="openvino_weight_format"):
        settings_factory(local_models__openvino_weight_format="int4")


def test_the_threads_per_core_default_to_2_and_read_their_variable(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QP_LOCAL_MODELS__THREADS_PER_CORE", raising=False)
    assert settings_factory().local_models.threads_per_core == 2

    monkeypatch.setenv("QP_LOCAL_MODELS__THREADS_PER_CORE", "1")
    assert settings_factory().local_models.threads_per_core == 1


def test_fewer_than_one_thread_per_core_is_refused(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match="threads_per_core"):
        settings_factory(local_models__threads_per_core=0)


def test_the_calibration_users_default_to_4_and_read_their_variable(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QP_CALIBRATION__USERS", raising=False)
    assert settings_factory().calibration.users == 4
    assert WorkflowSettings.variable_of("calibration", "users") == (
        "QP_CALIBRATION__USERS"
    )

    monkeypatch.setenv("QP_CALIBRATION__USERS", "8")
    assert settings_factory().calibration.users == 8


def test_fewer_than_one_calibration_user_is_refused(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match="users"):
        settings_factory(calibration__users=0)


def test_the_tests_settings_refuse_a_name_that_is_no_setting(
    settings_factory: SettingsFactory,
) -> None:
    # The settings ignore it: a test passing a removed name would run for real.
    with pytest.raises(TypeError, match="'vector_store_backend' is no setting"):
        settings_factory(vector_store_backend="fake")


def test_the_defaults_are_httpx_own() -> None:
    http = DefaultHttpSettings()

    pool = httpx.Limits(
        max_connections=http.max_connections,
        max_keepalive_connections=http.max_keepalive_connections,
        keepalive_expiry=http.keepalive_expiry_s,
    )
    assert pool == httpx.Limits(
        max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0
    )
    assert http.http2 is False


def test_the_pool_is_read_from_its_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_HTTP__MAX_CONNECTIONS", "8")
    monkeypatch.setenv("QP_HTTP__MAX_KEEPALIVE_CONNECTIONS", "0")
    monkeypatch.setenv("QP_HTTP__KEEPALIVE_EXPIRY_S", "1.5")
    monkeypatch.setenv("QP_HTTP__HTTP2", "true")

    http = settings_factory().http

    assert (http.max_connections, http.max_keepalive_connections) == (8, 0)
    assert (http.keepalive_expiry_s, http.http2) == (1.5, True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_connections", 0),
        ("max_keepalive_connections", -1),
        ("keepalive_expiry_s", -1),
    ],
)
def test_a_pool_value_out_of_range_is_refused(key: str, value: int) -> None:
    with pytest.raises(ValidationError, match=f"{key}"):
        DefaultHttpSettings.model_validate({key: value})


def test_a_components_http_block_refuses_a_pool_value(
    settings_factory: SettingsFactory,
) -> None:
    # One client is shared, so its pool is the top-level block's alone.
    with pytest.raises(ValidationError, match="max_connections"):
        settings_factory(components__llm__http={"max_connections": 8})
