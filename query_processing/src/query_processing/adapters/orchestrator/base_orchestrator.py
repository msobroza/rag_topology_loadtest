"""The orchestrator port: a library that runs the workflow graph along its edges.

It is shaped as a stage type is: its class is found in ``ORCHESTRATORS`` by
the name the settings give, asked before any component opens whether it runs
the graph, built once with what it runs, the opened graph, and has one method
that runs a request (ADR 0010). This module imports the
domain only: the application depends on it, and on nothing else of the
adapters.

Example:
    >>> @ORCHESTRATORS.register("mine")
    ... class MyOrchestrator(BaseOrchestrator):
    ...     @classmethod
    ...     def check_graph_runs(cls, graph): ...
    ...     def __init__(self, opened): ...
    ...     async def ainvoke(self, state): ...
"""

from __future__ import annotations

import abc
from typing import Any

from ...domain.graph_description import GraphDescription
from ...domain.opened_graph import OpenedGraph
from ...domain.registry import Registry


class BaseOrchestrator(abc.ABC):
    """A library that runs the workflow graph: built once, run for every request.

    For every graph its check accepts, an orchestrator gives the last state
    LangGraph gives.

    Args:
        opened: The stages, opened and timed, the edges and the conditional
            edges, the state's field types and its reducers.

    Example:
        >>> orchestrator = ORCHESTRATORS.get("langgraph")(opened)
        >>> final = await orchestrator.ainvoke(initial_state("q?", "query"))
    """

    @classmethod
    @abc.abstractmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        """Refuse a graph this library would not run as LangGraph does.

        Asked at start-up about the file's graph, once the toolkit's own checks
        accept it and before any component opens. Abstract, so an orchestrator
        states what it refuses, if anything.

        Args:
            graph: The workflow file's graph: its stages and edges.

        Raises:
            ValueError: Naming the stages, what the library would do, and what
                was expected.

        Example:
            >>> MyOrchestrator.check_graph_runs(settings.graph)
            ValueError: stage 'rerank' has two ways out at once, which ...
        """

    # Abstract, so an orchestrator states how it builds its library's graph
    # from the opened stages and edges (the class docstring holds the Args).
    @abc.abstractmethod
    def __init__(self, opened: OpenedGraph) -> None: ...

    @abc.abstractmethod
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run one request from its first state to its last.

        The first state is the application's (``initial_state``): every field
        that adds up (the metadata filter, the timings, the tokens) is already
        set, as its reducer would make it, so an orchestrator does not reduce
        the first state; it reduces only the stages' updates.

        Args:
            state: The state the request starts with.

        Returns:
            The state once the run reached ``END``.
        """


ORCHESTRATORS: Registry[type[BaseOrchestrator]] = Registry("orchestrator")
