"""Restart the process with another environment, started as it was started.

The dynamic linker reads ``LD_PRELOAD`` and the OpenMP runtime its variables
when a process starts, so a command that needs them replaces itself once with
an environment that holds them; the restarted process finds them there and
goes on (``config/apps/process_environment_settings.py`` works it out).

Example:
    >>> restart_with(None)  # nothing to change: returns, the process goes on
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence

Execve = Callable[[str, Sequence[str], Mapping[str, str]], None]


def restart_with(
    environment: Mapping[str, str] | None,
    argv: Sequence[str] | None = None,
    *,
    execve: Execve | None = None,
) -> None:
    """Replace the process with itself under ``environment``; with ``None``, return.

    Args:
        environment: The whole environment to restart with, or ``None`` when
            the process already holds what it needs.
        argv: The interpreter's own arguments, as ``sys.orig_argv`` holds
            them (the interpreter, its flags, then the console script or
            ``-m`` and the module); ``None`` reads ``sys.orig_argv``.
        execve: Replaces the process; ``os.execve``, looked up at the call so
            a test can record it, unless given.

    Example:
        >>> restart_with({"OMP_PROC_BIND": "close"})  # doctest: +SKIP
    """
    if environment is None:
        return
    # Not [sys.executable, *sys.argv]: under "python -m" sys.argv[0] is the
    # module's file, which run as a script loses its package (relative imports
    # break), and sys.argv holds no interpreter flag.
    arguments = list(sys.orig_argv if argv is None else argv)
    # execve drops what Python has buffered but not yet written.
    sys.stdout.flush()
    sys.stderr.flush()
    (execve or os.execve)(sys.executable, arguments, environment)
