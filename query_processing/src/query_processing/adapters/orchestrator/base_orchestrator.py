"""The orchestrator port: a library that runs the workflow graph along its edges.

It is shaped as a stage type is: its class is found in ``ORCHESTRATORS`` by
the name the settings give, built once with what it runs, the opened graph,
and has one method that runs a request (ADR 0010). This module imports the
domain only: the application depends on it, and on nothing else of the
adapters.

Example:
    >>> @ORCHESTRATORS.register("mine")
    ... class MyOrchestrator(BaseOrchestrator):
    ...     def __init__(self, opened): ...
    ...     async def ainvoke(self, state): ...
"""

from __future__ import annotations

import abc
from typing import Any

from ...domain.opened_graph import OpenedGraph
from ...domain.registry import Registry


class BaseOrchestrator(abc.ABC):
    """A library that runs the workflow graph: built once, run for every request.

    Args:
        opened: The stages, opened and timed, the edges and the conditional
            edges, the state's field types and its reducers.

    Example:
        >>> orchestrator = ORCHESTRATORS.get("langgraph")(opened)
        >>> final = await orchestrator.ainvoke(initial_state("q?", "query"))
    """

    # Abstract, so an orchestrator states how it builds its library's graph
    # from the opened stages and edges (the class docstring holds the Args).
    @abc.abstractmethod
    def __init__(self, opened: OpenedGraph) -> None: ...

    @abc.abstractmethod
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run one request from its first state to its last.

        Args:
            state: The state the request starts with.

        Returns:
            The state once the run reached ``END``.
        """


ORCHESTRATORS: Registry[type[BaseOrchestrator]] = Registry("orchestrator")
