"""``OvmsSettings``: an OVMS App's own streams, threads and start timeout.

They are no value of the workflow file: one OVMS App serves one component, so
they are that App's flat ``QP_OVMS_*`` variables. The calibration users, which
every model server's calibration shares, are tested with the shared blocks
(``test_shared_settings.py``).
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from query_processing.config.ovms_settings import OvmsSettings
from rag_components import shell_settings


@pytest.fixture(autouse=True)
def _no_ovms_variable_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in shell_settings(os.environ, ("QP_OVMS_",)):
        monkeypatch.delenv(name)


def ovms_settings(**values: object) -> OvmsSettings:
    """The settings without reading a stray ``.env``."""
    return OvmsSettings(_env_file=None, **values)  # type: ignore[arg-type]


def test_ovms_is_calibrated_by_default() -> None:
    s = ovms_settings()

    assert (s.num_streams, s.threads) == (0, 0)
    assert s.start_timeout_s == 120.0


def test_the_variables_are_named_as_qp_calibrate_prints_them() -> None:
    assert OvmsSettings.variable_of("num_streams") == "QP_OVMS_NUM_STREAMS"
    assert OvmsSettings.variable_of("threads") == "QP_OVMS_THREADS"
    assert OvmsSettings.variable_of("start_timeout_s") == "QP_OVMS_START_TIMEOUT_S"


def test_the_ovms_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_OVMS_NUM_STREAMS", "2")
    monkeypatch.setenv("QP_OVMS_THREADS", "4")
    monkeypatch.setenv("QP_OVMS_START_TIMEOUT_S", "30")

    s = ovms_settings()

    assert (s.num_streams, s.threads, s.start_timeout_s) == (2, 4, 30.0)


@pytest.mark.parametrize(
    "overrides",
    [{"num_streams": -1}, {"threads": -1}, {"start_timeout_s": 0}],
)
def test_out_of_range_values_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ovms_settings(**overrides)


def test_a_pinned_stream_count_above_the_pinned_threads_is_refused() -> None:
    with pytest.raises(ValidationError) as raised:
        ovms_settings(num_streams=4, threads=2)

    message = str(raised.value)
    assert "QP_OVMS_NUM_STREAMS=4" in message
    assert "QP_OVMS_THREADS=2" in message


@pytest.mark.parametrize(("streams", "threads"), [(4, 0), (0, 1), (2, 2), (1, 4)])
def test_streams_are_checked_against_threads_only_when_both_are_pinned(
    streams: int, threads: int
) -> None:
    s = ovms_settings(num_streams=streams, threads=threads)

    assert (s.num_streams, s.threads) == (streams, threads)


# --- the LLM's two batching options (OVMS's own flags) -------------------------------


def test_the_llms_batching_options_keep_ovmss_defaults() -> None:
    s = ovms_settings()

    assert (s.dynamic_split_fuse, s.max_batched_tokens) == (True, 0)
    assert s.llm_flags() == []  # at OVMS's defaults, no flag is passed


def test_the_llms_batching_options_come_from_their_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_OVMS_DYNAMIC_SPLIT_FUSE", "false")
    monkeypatch.setenv("QP_OVMS_MAX_BATCHED_TOKENS", "8192")

    s = ovms_settings()

    assert s.llm_flags() == [
        "--dynamic_split_fuse",
        "false",
        "--max_num_batched_tokens",
        "8192",
    ]


def test_a_negative_batched_token_count_is_refused() -> None:
    with pytest.raises(ValidationError):
        ovms_settings(max_batched_tokens=-1)


@pytest.mark.parametrize(
    ("variable", "value"),
    [("QP_OVMS_DYNAMIC_SPLIT_FUSE", "false"), ("QP_OVMS_MAX_BATCHED_TOKENS", "512")],
)
def test_an_llm_option_on_an_embedder_or_reranker_app_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match="applies to an LLM") as raised:
        ovms_settings().require_no_llm_option("reranker")

    message = str(raised.value)
    assert variable in message
    assert "applies to an LLM" in message
    assert "reranker" in message


def test_no_llm_option_set_lets_an_embedder_app_start() -> None:
    ovms_settings().require_no_llm_option("embedder")


@pytest.mark.parametrize("variable", ["QP_OVMS_NUM_STREAMS", "QP_OVMS_THREADS"])
def test_a_pinned_stream_or_thread_count_on_an_llm_app_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "2")

    with pytest.raises(ValueError, match="serves it uncalibrated") as raised:
        ovms_settings().require_no_stream_pin("llm")

    message = str(raised.value)
    assert variable in message
    assert "llm" in message


def test_no_stream_pin_lets_an_llm_app_start() -> None:
    ovms_settings().require_no_stream_pin("llm")
