"""One OpenVINO Model Server process on a free local port, for an ``async with``.

OVMS takes its settings only when it starts, so calibrating it means starting it
once per candidate. On entry the process starts on 127.0.0.1 and a port the
operating system picks, and the ``ready`` callable given is polled until the
server answers; a process that exits meanwhile fails at once with its exit code,
one that never answers fails after ``start_timeout_s``. On exit it is terminated,
then killed if it does not stop in time. Its output goes to our standard error:
our standard output may be what a command prints for a script to read.

Example:
    >>> async with OvmsProcess(config, ready=answers, start_timeout_s=120.0) as ovms:
    ...     await call(f"http://127.0.0.1:{ovms.port}/v3/rerank")
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import TracebackType

Ready = Callable[[int], Awaitable[bool]]  # given the port: does the server answer?
_STDERR_FD = 2  # a descriptor, not sys.stderr: a test's captured stream has none


class OvmsStartError(RuntimeError):
    """OVMS exited, or did not answer, before it was ready."""


def free_local_port() -> int:
    """Return a TCP port on 127.0.0.1 that nothing listens on right now.

    The operating system picks it (bind port 0); the socket is closed before
    returning, so another process could take it first, which OVMS then reports
    as an exit.

    Returns:
        The port number.

    Example:
        >>> 1024 <= free_local_port() <= 65535
        True
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class OvmsProcess:
    """An ``ovms`` process serving ``config_path`` while the ``async with`` lasts.

    Args:
        config_path: The config file OVMS serves (``--config_path``).
        ready: Called with the port until it returns True: one real request.
        start_timeout_s: How long the server may take to answer.
        binary: The OVMS executable, looked up on ``PATH`` unless a path.
        poll_s: The wait between two ``ready`` calls.
        stop_timeout_s: How long a terminated process may take to exit before
            it is killed.
        server_flags: The App's other ``ovms`` flags
            (``OvmsSettings.server_flags()``), after the config, so a
            calibration candidate is timed as it will be served.

    Example:
        >>> async with OvmsProcess(config, ready=answers, start_timeout_s=60) as ovms:
        ...     ovms.port
        38211
    """

    def __init__(  # noqa: PLR0913 - every one but the config is a keyword
        self,
        config_path: Path,
        *,
        ready: Ready,
        start_timeout_s: float,
        binary: str = "ovms",
        poll_s: float = 0.2,
        stop_timeout_s: float = 10.0,
        server_flags: Sequence[str] = (),
    ) -> None:
        self._config_path, self._ready = config_path, ready
        self._server_flags = tuple(server_flags)
        self._start_timeout_s, self._binary = start_timeout_s, binary
        self._poll_s, self._stop_timeout_s = poll_s, stop_timeout_s
        self._process: asyncio.subprocess.Process | None = None
        self.port = 0  # set on entry

    async def __aenter__(self) -> OvmsProcess:
        self.port = free_local_port()
        self._process = await asyncio.create_subprocess_exec(
            self._binary,
            *("--rest_bind_address", "127.0.0.1", "--rest_port", str(self.port)),
            *("--config_path", str(self._config_path)),
            *self._server_flags,
            stdout=_STDERR_FD,
            stderr=_STDERR_FD,
        )
        try:
            await self._wait_until_ready(self._process)
        except BaseException:
            await self._stop(self._process)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._process is not None:
            await self._stop(self._process)

    async def _wait_until_ready(self, process: asyncio.subprocess.Process) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._start_timeout_s
        while True:
            self._raise_if_exited(process)
            if await self._ready(self.port):
                return
            if loop.time() >= deadline:
                raise OvmsStartError(
                    f"{self._binary} did not answer on 127.0.0.1:{self.port} within "
                    f"{self._start_timeout_s:g} s (config {self._config_path}); "
                    "its log is above, on standard error"
                )
            await asyncio.sleep(self._poll_s)

    def _raise_if_exited(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            return
        raise OvmsStartError(
            f"{self._binary} exited with code {process.returncode} before it "
            f"answered on 127.0.0.1:{self.port} (config {self._config_path}); its "
            "log is above, on standard error"
        )

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), self._stop_timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
