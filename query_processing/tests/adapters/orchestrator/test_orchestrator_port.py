"""The orchestrator port's promise, held for every registered orchestrator.

For every graph its class-level check accepts, an orchestrator gives the last
state LangGraph gives (ADR 0010). On each graph of one set, every registered
orchestrator either refuses it with a ``ValueError`` or ends each request in
LangGraph's state; a stage's ``LookupError`` reaches the workflow's run as
itself; and concurrent runs of one built graph each end in their own state.
The orchestrators are the registry's, read when the tests are collected
(``registered_orchestrators``), so a library added is tested with no edit.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, NamedTuple

import pytest

from query_processing.adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from query_processing.adapters.orchestrator.langgraph.langgraph_orchestrator import (
    LangGraphOrchestrator,
)
from query_processing.application.workflow import CompiledWorkflow
from query_processing.application.workflow_state import WorkflowState, initial_state
from query_processing.testing import registered_orchestrators
from workflow_fakes import asking, described, opened_of, said

QUERY = initial_state("q?", "query")
RETRIEVE = initial_state("q?", "retrieve")
WITH_HISTORY = initial_state("q?", "query", conversational_history=[{"x": 1}])


class GraphCase(NamedTuple):
    """A graph block, and the first states of the requests it is run with."""

    graph: dict[str, Any]
    requests: list[WorkflowState]


GRAPHS = {
    "chain": GraphCase(
        {
            "stages": {
                "a": said(["conversational_query"], ["draft"]),
                "b": said(["draft"], ["note"]),
                "c": said(["note"], ["answer"]),
            },
            "edges": [["START", "a", "b", "c", "END"]],
        },
        [QUERY],
    ),
    "yes-or-no-branch": GraphCase(
        {
            "stages": {
                "draft": said(["conversational_query"], ["draft"]),
                "polish": said(["draft"], ["answer"]),
            },
            "edges": [["START", "draft"], ["polish", "END"]],
            "conditional_edges": [
                asking(
                    "draft", "is_query_mode", ["mode"], {True: "polish", False: "END"}
                )
            ],
        },
        [QUERY, RETRIEVE],
    ),
    "label-branch": GraphCase(
        {
            "stages": {
                "route": said(["conversational_query"], ["route"]),
                "embed": said(["conversational_query"], ["query_embedding"]),
                "rewrite": said(["conversational_history"], ["standalone_query"]),
            },
            "edges": [["START", "route"], ["embed", "END"], ["rewrite", "END"]],
            "conditional_edges": [
                asking(
                    "route",
                    "label",
                    ["route"],
                    {"self_contained": "embed", "leans_on_history": "rewrite"},
                )
            ],
        },
        [
            {**QUERY, "wanted_route": "self_contained"},
            {**QUERY, "wanted_route": "leans_on_history"},
        ],
    ),
    "conditional-edge-at-start": GraphCase(
        {
            "stages": {
                "rewrite": said(["conversational_history"], ["standalone_query"]),
                "embed": said(["standalone_query"], ["query_embedding"]),
            },
            "edges": [["rewrite", "embed", "END"]],
            "conditional_edges": [
                asking(
                    "START",
                    "has_history",
                    ["conversational_history"],
                    {True: "rewrite", False: "embed"},
                )
            ],
        },
        [QUERY, WITH_HISTORY],
    ),
    "two-plain-edges-out-of-start": GraphCase(
        {
            "stages": {
                "a": said(["conversational_query"], ["draft"]),
                "b": said(["conversational_query"], ["note"]),
            },
            "edges": [["START", "a", "END"], ["START", "b", "END"]],
        },
        [QUERY],
    ),
    "two-conditional-edges-on-one-source": GraphCase(
        {
            "stages": {
                "a": said(["conversational_query"], ["draft"]),
                "b": said(["draft"], ["note"]),
                "c": said(["draft"], ["answer"]),
            },
            "edges": [["START", "a"], ["b", "END"], ["c", "END"]],
            "conditional_edges": [
                asking("a", "is_query_mode", ["mode"], {True: "b", False: "END"}),
                asking(
                    "a",
                    "has_history",
                    ["conversational_history"],
                    {True: "c", False: "END"},
                ),
            ],
        },
        [QUERY, RETRIEVE, WITH_HISTORY],
    ),
    # polish writes the mode, so a query runs draft, polish, then draft again.
    "loop": GraphCase(
        {
            "stages": {
                "draft": said(["conversational_query"], ["draft"]),
                "polish": said(["draft"], ["mode"]),
            },
            "edges": [["START", "draft"], ["polish", "draft"]],
            "conditional_edges": [
                asking(
                    "draft", "is_query_mode", ["mode"], {True: "polish", False: "END"}
                )
            ],
        },
        [QUERY, RETRIEVE],
    ),
}


def refuses(orchestrator: str, graph: dict[str, Any]) -> bool:
    """Whether the orchestrator's class-level check refuses ``graph``."""
    try:
        ORCHESTRATORS.get(orchestrator).check_graph_runs(described(graph))
    except ValueError:
        return True
    return False


@pytest.mark.parametrize("orchestrator", registered_orchestrators())
@pytest.mark.parametrize("case", sorted(GRAPHS))
async def test_on_each_graph_it_refuses_or_ends_in_the_state_langgraph_gives(
    orchestrator: str, case: str
) -> None:
    graph, requests = GRAPHS[case]
    if refuses(orchestrator, graph):
        return  # the promise holds only for a graph its check accepts
    opened = opened_of(described(graph))

    for first in requests:
        expected = await LangGraphOrchestrator(opened).ainvoke(dict(first))
        final = await ORCHESTRATORS.get(orchestrator)(opened).ainvoke(dict(first))

        assert final == expected, first
        assert list(final["timings_ms"]) == list(expected["timings_ms"])
        assert list(final["tokens"]) == list(expected["tokens"])


async def test_the_loop_runs_its_stages_again_on_langgraph() -> None:
    # Without this, the loop case could pass with a loop that never loops.
    opened = opened_of(described(GRAPHS["loop"].graph))

    final = await LangGraphOrchestrator(opened).ainvoke(dict(QUERY))

    assert final["timings_ms"] == {"draft": 2.0, "polish": 1.0}


async def failing_stage(state: dict[str, Any]) -> dict[str, Any]:
    raise LookupError("no passage")


@pytest.mark.parametrize("orchestrator", registered_orchestrators())
async def test_a_stage_s_lookup_error_reaches_the_workflow_s_run_as_itself(
    orchestrator: str,
) -> None:
    opened = opened_of(described(GRAPHS["chain"].graph))
    failing = dataclasses.replace(opened, stages={**opened.stages, "b": failing_stage})
    workflow = CompiledWorkflow(
        ORCHESTRATORS.get(orchestrator)(failing), stage_names=list(failing.stages)
    )

    with pytest.raises(LookupError, match="no passage"):
        await workflow.run("q?")


@pytest.mark.parametrize("orchestrator", registered_orchestrators())
async def test_concurrent_runs_of_one_built_graph_each_end_in_their_own_state(
    orchestrator: str,
) -> None:
    opened = opened_of(described(GRAPHS["yes-or-no-branch"].graph))
    built = ORCHESTRATORS.get(orchestrator)(opened)
    # Every other request a retrieve, which ends before polish.
    firsts = [
        initial_state(f"q{n}?", "query" if n % 2 == 0 else "retrieve")
        for n in range(200)
    ]

    finals = await asyncio.gather(*(built.ainvoke(first) for first in firsts))

    assert len(finals) == 200
    assert [f["conversational_query"] for f in finals] == [f"q{n}?" for n in range(200)]
    ran = [list(final["timings_ms"]) for final in finals]
    assert ran == [["draft", "polish"] if n % 2 == 0 else ["draft"] for n in range(200)]
