"""How a command reports a failure an operator can act on: one line, exit code 1.

A command whose standard output is read by a script (``qp-calibrate``,
``qp-ovms-config``) keeps that output clean and says what went wrong on one
line of standard error instead of a traceback.

Example:
    >>> return one_line_failure("qp-calibrate", "cannot read the settings", exc)
    1
"""

from __future__ import annotations

import sys


def one_line_failure(prog: str, what: str, exc: BaseException) -> int:
    """Print ``<prog>: <what>: <error>`` on one line of standard error.

    Args:
        prog: The command's name.
        what: What it could not do.
        exc: Why; a message over several lines (pydantic's) is joined into one.

    Returns:
        1, the exit code of such a failure.

    Example:
        >>> one_line_failure("qp-ovms-config", "cannot serve reranker", OSError("x"))
        1
    """
    # One line: pydantic's message spans several, and the value it names is on one.
    detail = " ".join(str(exc).split())
    print(f"{prog}: {what}: {detail}", file=sys.stderr)
    return 1
