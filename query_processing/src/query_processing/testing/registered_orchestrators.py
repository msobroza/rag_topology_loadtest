"""The orchestrators a test runs on: every name the registry holds, read once.

A test parametrized over them holds every library to the port's promise (ADR
0010), and one added under ``adapters/orchestrator/`` is tested with no test
edited. The names are read on the first call, when the tests are collected,
and kept: an orchestrator a test registers for itself, while it runs, is
never among them.

Example:
    >>> @pytest.mark.parametrize("orchestrator", registered_orchestrators())
    ... def test_it_runs(orchestrator: str) -> None: ...
"""

from __future__ import annotations

import functools

from ..adapters.orchestrator.base_orchestrator import ORCHESTRATORS


# Cached so that a call made while a test has registered an orchestrator of
# its own still gives the names the tests were collected with.
@functools.cache
def registered_orchestrators() -> tuple[str, ...]:
    """Return the name of every registered orchestrator, sorted, as first read.

    Returns:
        The names ``ORCHESTRATORS`` held at the first call.

    Example:
        >>> registered_orchestrators()
        ('langgraph', 'pydantic_graph')
    """
    return ORCHESTRATORS.names()
