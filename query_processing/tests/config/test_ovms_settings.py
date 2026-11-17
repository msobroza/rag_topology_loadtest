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
