"""``OvmsProcess`` against a stand-in ``ovms``: started, waited for, stopped.

Every wait is bounded (start timeouts of a few seconds, ``asyncio.timeout`` on
each test body) so a broken stand-in cannot hang the suite.
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from ovms_stand_ins import ovms_starts, write_ovms_export, write_ovms_stand_in
from query_processing.infrastructure.ovms.ovms_process import (
    OvmsProcess,
    OvmsStartError,
    free_local_port,
)

BOUND_S = 30.0  # no test here may take longer


async def listens(port: int) -> bool:
    """Whether something accepts a connection on 127.0.0.1:port."""
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


@pytest.fixture
def config(tmp_path: Path) -> Path:
    write_ovms_export(tmp_path / "models", "reranker", "rr")
    return tmp_path / "models" / "config_reranker.json"


def stand_in(tmp_path: Path, behaviour: str = "answers") -> str:
    bin_dir = write_ovms_stand_in(
        tmp_path / "bin", tmp_path / "ovms.jsonl", behaviour=behaviour
    )
    return str(bin_dir / "ovms")


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_free_port_can_be_bound() -> None:
    port = free_local_port()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


async def test_it_starts_ovms_on_a_free_local_port_and_stops_it(
    tmp_path: Path, config: Path
) -> None:
    async with asyncio.timeout(BOUND_S):
        process = OvmsProcess(
            config, ready=listens, start_timeout_s=10.0, binary=stand_in(tmp_path)
        )
        async with process as ovms:
            assert await listens(ovms.port)
            pid = ovms._process.pid  # type: ignore[union-attr]
        assert not await listens(ovms.port)

    [start] = ovms_starts(tmp_path / "ovms.jsonl")
    assert start["argv"] == [
        "--rest_bind_address",
        "127.0.0.1",
        "--rest_port",
        str(ovms.port),
        "--config_path",
        str(config),
    ]
    assert not is_running(pid)


async def test_its_output_goes_to_standard_error_not_standard_output(
    tmp_path: Path, config: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    async with asyncio.timeout(BOUND_S):
        binary = stand_in(tmp_path)
        async with OvmsProcess(
            config, ready=listens, start_timeout_s=10.0, binary=binary
        ):
            pass

    captured = capfd.readouterr()
    assert "stand-in ovms: starting" in captured.err
    assert "stand-in ovms" not in captured.out


async def test_a_process_that_exits_fails_at_once_with_its_exit_code(
    tmp_path: Path, config: Path
) -> None:
    binary = stand_in(tmp_path, "exits")
    process = OvmsProcess(config, ready=listens, start_timeout_s=60.0, binary=binary)

    async with asyncio.timeout(BOUND_S):
        with pytest.raises(OvmsStartError, match="exited with code 3"):
            await process.__aenter__()


async def test_a_server_that_never_answers_fails_after_the_timeout_and_is_stopped(
    tmp_path: Path, config: Path
) -> None:
    binary = stand_in(tmp_path, "silent")
    process = OvmsProcess(config, ready=listens, start_timeout_s=1.0, binary=binary)

    async with asyncio.timeout(BOUND_S):
        with pytest.raises(OvmsStartError, match=r"did not answer .* within 1 s"):
            await process.__aenter__()
    pid = process._process.pid  # type: ignore[union-attr]
    assert not is_running(pid)


async def test_a_process_that_ignores_terminate_is_killed(
    tmp_path: Path, config: Path
) -> None:
    binary = stand_in(tmp_path, "stubborn")
    process = OvmsProcess(
        config,
        ready=listens,
        start_timeout_s=10.0,
        binary=binary,
        stop_timeout_s=0.5,
    )

    async with asyncio.timeout(BOUND_S):
        async with process as ovms:
            port = ovms.port
    assert not await listens(port)


async def test_a_missing_binary_is_an_os_error(tmp_path: Path, config: Path) -> None:
    binary = str(tmp_path / "no-such-ovms")
    process = OvmsProcess(config, ready=listens, start_timeout_s=1.0, binary=binary)

    with pytest.raises(FileNotFoundError):
        await process.__aenter__()
