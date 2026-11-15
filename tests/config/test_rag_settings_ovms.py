"""``RagSettings``: the OVMS block and the calibration users."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.rag_settings import RagSettings

SettingsFactory = Callable[..., RagSettings]


def test_ovms_is_calibrated_by_default(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert (s.ovms_num_streams, s.ovms_threads) == (0, 0)
    assert s.ovms_start_timeout_s == 120.0
    assert s.calibration_users == 4


def test_the_variables_are_named_as_rag_calibrate_prints_them() -> None:
    assert RagSettings.variable_of("ovms_num_streams") == "RAG_OVMS_NUM_STREAMS"
    assert RagSettings.variable_of("ovms_threads") == "RAG_OVMS_THREADS"


def test_the_ovms_settings_come_from_the_environment(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_OVMS_NUM_STREAMS", "2")
    monkeypatch.setenv("RAG_OVMS_THREADS", "4")
    monkeypatch.setenv("RAG_CALIBRATION_USERS", "8")

    s = settings_factory()

    assert (s.ovms_num_streams, s.ovms_threads, s.calibration_users) == (2, 4, 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ovms_num_streams": -1},
        {"ovms_threads": -1},
        {"ovms_start_timeout_s": 0},
        {"calibration_users": 0},
    ],
)
def test_out_of_range_values_are_refused(
    settings_factory: SettingsFactory, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        settings_factory(**overrides)


def test_a_pinned_stream_count_above_the_pinned_threads_is_refused(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError) as raised:
        settings_factory(ovms_num_streams=4, ovms_threads=2)

    message = str(raised.value)
    assert "RAG_OVMS_NUM_STREAMS=4" in message
    assert "RAG_OVMS_THREADS=2" in message


@pytest.mark.parametrize(("streams", "threads"), [(4, 0), (0, 1), (2, 2), (1, 4)])
def test_streams_are_checked_against_threads_only_when_both_are_pinned(
    settings_factory: SettingsFactory, streams: int, threads: int
) -> None:
    s = settings_factory(ovms_num_streams=streams, ovms_threads=threads)

    assert (s.ovms_num_streams, s.ovms_threads) == (streams, threads)
