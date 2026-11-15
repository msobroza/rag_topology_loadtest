"""What the components share: four top-level blocks of the workflow file.

Each block is read from the workflow file and from its environment path, a
variable wins over the file key by key, and a key a block does not have is
refused. The flat variables of before are read no more.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_graphs import fixture_graph, write_workflow_file
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.shared_settings import (
    CalibrationSettings,
    DefaultHttpSettings,
    DominoSettings,
    LocalModelsSettings,
)
from rag_load_test.config.workflow_file import WORKFLOW_FILE_VARIABLE

SettingsFactory = Callable[..., RagSettings]
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
        runtime="openvino", openvino_dir="models/openvino", workers=4, cpus=0
    )
    assert s.calibration == CalibrationSettings(budget_s=90.0, rounds=1)
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
    monkeypatch.setenv("RAG_LOCAL_MODELS__CPUS", "8")
    monkeypatch.setenv("RAG_HTTP__RETRIES", "0")
    monkeypatch.setenv("RAG_DOMINO__TRACING", "false")
    monkeypatch.setenv("RAG_CALIBRATION__ROUNDS", "5")

    s = settings_factory()

    assert (s.local_models.cpus, s.local_models.runtime) == (8, "torch")
    assert (s.http.retries, s.http.timeout_s) == (0, 9)
    assert (s.domino.tracing, s.domino.token_ttl_s) == (False, 60)
    assert (s.calibration.rounds, s.calibration.budget_s) == (5, 12.5)


def test_each_shared_value_names_the_variable_it_is_read_from() -> None:
    paths = {
        ("local_models", "runtime"): "RAG_LOCAL_MODELS__RUNTIME",
        ("local_models", "openvino_dir"): "RAG_LOCAL_MODELS__OPENVINO_DIR",
        ("calibration", "budget_s"): "RAG_CALIBRATION__BUDGET_S",
        ("http", "backoff_s"): "RAG_HTTP__BACKOFF_S",
        ("domino", "token_url"): "RAG_DOMINO__TOKEN_URL",
    }

    assert {path: RagSettings.variable_of(*path) for path in paths} == paths


@pytest.mark.parametrize(
    ("variable", "key"),
    [
        ("RAG_LOCAL_MODELS__MODELS_DIR", "models_dir"),
        ("RAG_HTTP__TIMEOUT", "timeout"),
        ("RAG_DOMINO__TOKEN_TTL", "token_ttl"),
        ("RAG_CALIBRATION__BUDGET", "budget"),
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
        "RAG_VECTOR_STORE_BACKEND": "fake",
        "RAG_ES_URL": "http://elsewhere:9200",
        "RAG_CHROMA_PATH": "elsewhere",
        "RAG_FAKE_STORE_SEED_DOCUMENTS": "3",
        "RAG_LOCAL_MODEL_RUNTIME": "torch",
        "RAG_OPENVINO_MODELS_DIR": "elsewhere",
        "RAG_MODEL_WORKERS": "1",
        "RAG_MODEL_CPUS": "1",
        "RAG_CALIBRATION_ROUNDS": "9",
        "RAG_HTTP_RETRIES": "0",
        "RAG_DOMINO_TRACING": "true",
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


def test_the_tests_settings_refuse_a_name_that_is_no_setting(
    settings_factory: SettingsFactory,
) -> None:
    # The settings ignore it: a test passing a removed name would run for real.
    with pytest.raises(TypeError, match="'vector_store_backend' is no setting"):
        settings_factory(vector_store_backend="fake")
