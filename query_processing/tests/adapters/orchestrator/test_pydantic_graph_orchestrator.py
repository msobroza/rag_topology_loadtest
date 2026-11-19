"""The pydantic-graph orchestrator: the state LangGraph gives, and what it refuses.

Through the port: an opened graph of stand-in stages, each writing what its
description says, is run on this orchestrator and on LangGraph's, and the
last states are equal. Its class refuses the two shapes its library does not
run as LangGraph does, more than one way out at once and a loop, naming the
stages, and accepts the fixture workflow's graph.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import pytest

from query_processing.adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from query_processing.adapters.orchestrator.langgraph.langgraph_orchestrator import (
    LangGraphOrchestrator,
)
from query_processing.adapters.orchestrator.pydantic_graph.pydantic_graph_orchestrator import (
    PydanticGraphOrchestrator,
)
from query_processing.application.state_fields import (
    STATE_REDUCERS,
    state_field_types,
)
from query_processing.application.workflow_state import initial_state
from query_processing.domain.graph_description import ConditionalEdge
from query_processing.domain.opened_graph import OpenedConditionalEdge, OpenedGraph
from rag_graphs import fixture_graph, rag_graph
from workflow_fakes import (
    ON_LEAVE,
    ON_PAY,
    asking,
    described,
    opened_of,
    said,
    stand_in,
)

# The RAG's shape: a conditional edge at START, a label branch, a yes-or-no
# branch, and paths that meet when only one of them is taken.
ROUTED_SHAPE = {
    "stages": {
        "embed_for_route": said(["conversational_query"], ["routing_embedding"]),
        "route": said(["routing_embedding"], ["route"]),
        "rewrite": said(["conversational_history"], ["standalone_query"]),
        "embed": said(["standalone_query"], ["query_embedding"]),
        "retrieve": said(["query_embedding"], ["candidates", "metadata_filter"]),
        "rerank": said(["candidates"], ["reranked"]),
        "generate": said(["reranked"], ["answer"]),
    },
    "edges": [
        ["embed_for_route", "route"],
        ["rewrite", "embed", "retrieve", "rerank"],
        ["generate", "END"],
    ],
    "conditional_edges": [
        asking(
            "START",
            "has_history",
            ["conversational_history"],
            {True: "embed_for_route", False: "embed"},
        ),
        asking(
            "route",
            "label",
            ["route"],
            {"self_contained": "embed", "leans_on_history": "rewrite"},
        ),
        asking("rerank", "is_query_mode", ["mode"], {True: "generate", False: "END"}),
    ],
}
# The first state of each kind of request, on the RAG's shape.
REQUESTS = {
    "first message": initial_state("q?", "query", metadata_filter=[ON_PAY]),
    "self-contained": {
        **initial_state("q?", "query", conversational_history=[{"x": 1}]),
        "wanted_route": "self_contained",
    },
    "leans on history": {
        **initial_state("q?", "query", conversational_history=[{"x": 1}]),
        "wanted_route": "leans_on_history",
    },
    "retrieve": initial_state("q?", "retrieve", metadata_filter=[ON_PAY, ON_LEAVE]),
}


def test_it_registers_as_pydantic_graph() -> None:
    assert ORCHESTRATORS.get("pydantic_graph") is PydanticGraphOrchestrator


@pytest.mark.parametrize("request_name", sorted(REQUESTS))
async def test_a_request_ends_in_the_state_langgraph_gives(request_name: str) -> None:
    opened = opened_of(described(ROUTED_SHAPE))
    first = REQUESTS[request_name]

    expected = await LangGraphOrchestrator(opened).ainvoke(dict(first))
    final = await PydanticGraphOrchestrator(opened).ainvoke(dict(first))

    assert final == expected
    assert list(final["timings_ms"]) == list(expected["timings_ms"])
    assert list(final["tokens"]) == list(expected["tokens"])


async def test_the_fixture_graph_ends_in_the_state_langgraph_gives() -> None:
    opened = opened_of(rag_graph("hybrid"))
    first = initial_state("q?", "query")

    expected = await LangGraphOrchestrator(opened).ainvoke(dict(first))
    final = await PydanticGraphOrchestrator(opened).ainvoke(dict(first))

    assert final == expected
    assert list(final["timings_ms"]) == ["embed", "retrieve", "rerank", "generate"]


async def test_the_first_state_handed_in_is_not_changed() -> None:
    first = initial_state("q?", "query")
    kept = dict(first)

    await PydanticGraphOrchestrator(opened_of(rag_graph())).ainvoke(first)

    assert first == kept


def test_a_conditional_edge_is_a_step_named_after_its_source_and_edge_condition() -> (
    None
):
    orchestrator = PydanticGraphOrchestrator(opened_of(described(ROUTED_SHAPE)))

    steps = set(orchestrator.graph.nodes)

    assert {"START:has_history", "route:label", "rerank:is_query_mode"} <= steps
    assert set(ROUTED_SHAPE["stages"]) <= steps


def answering(answer: object) -> Callable[[dict[str, Any]], Any]:
    """An edge condition asked of the state that answers ``answer``, unchecked."""
    return lambda state: answer


def yes_or_no_after(ask: Callable[[dict[str, Any]], Any]) -> OpenedGraph:
    """START, then ``draft``, then a yes-or-no branch: yes to ``polish``, no to END."""
    edge = ConditionalEdge.model_validate(
        asking("draft", "is_long", ["draft"], {True: "polish", False: "END"})
    )
    return OpenedGraph(
        stages={
            "draft": stand_in("draft", ["draft"]),
            "polish": stand_in("polish", []),
        },
        edges=[("START", "draft"), ("polish", "END")],
        conditional_edges=[OpenedConditionalEdge(edge, ask)],
        field_types=state_field_types({"draft": str}),
        reducers=STATE_REDUCERS,
    )


async def test_the_yes_row_is_taken_on_an_answer_of_true() -> None:
    final = await PydanticGraphOrchestrator(yes_or_no_after(answering(True))).ainvoke(
        initial_state("q?", "query")
    )

    assert list(final["timings_ms"]) == ["draft", "polish"]


class Label(str):
    """A ``str`` subclass, as a classifier's ``numpy.str_`` label is."""


async def test_a_label_of_a_str_subclass_takes_its_str_row() -> None:
    edge = ConditionalEdge.model_validate(
        asking("draft", "label", ["draft"], {"long": "polish", "short": "END"})
    )
    opened = dataclasses.replace(
        yes_or_no_after(answering(True)),
        conditional_edges=[OpenedConditionalEdge(edge, answering(Label("long")))],
    )

    final = await PydanticGraphOrchestrator(opened).ainvoke(
        initial_state("q?", "query")
    )

    assert list(final["timings_ms"]) == ["draft", "polish"]


async def test_an_answer_of_1_does_not_take_the_true_row() -> None:
    orchestrator = PydanticGraphOrchestrator(yes_or_no_after(answering(1)))

    # 1 == True in Python: a row matched on equality alone would run polish.
    with pytest.raises(RuntimeError, match="No branch matched"):
        await orchestrator.ainvoke(initial_state("q?", "query"))


@pytest.mark.parametrize(
    "graph",
    [
        fixture_graph(),
        *(rag_graph(mode).model_dump() for mode in ("dense", "lexical", "hybrid")),
        ROUTED_SHAPE,
    ],
    ids=["fixture", "dense", "lexical", "hybrid", "routed-shape"],
)
def test_it_accepts_one_way_out_of_each_stage_at_a_time(graph: dict[str, Any]) -> None:
    PydanticGraphOrchestrator.check_graph_runs(described(graph))


SEVERAL_WAYS_OUT = (
    "{names} {have} more than one way out at once; pydantic-graph ends a run when "
    "the first reaches END and cancels the others (expected one plain edge or one "
    "conditional edge out of each stage)"
)
TWO_PLAIN_EDGES_OUT_OF_START = {
    "stages": {"a": said(["conversational_query"], ["x"]), "b": said(["x"], ["y"])},
    "edges": [["START", "a", "END"], ["START", "b", "END"]],
}
TWO_PLAIN_EDGES_OUT_OF_A_STAGE = {
    "stages": {
        "a": said(["conversational_query"], ["x"]),
        "b": said(["x"], ["y"]),
        "c": said(["x"], ["z"]),
    },
    "edges": [["START", "a", "b", "END"], ["a", "c", "END"]],
}
A_PLAIN_AND_A_CONDITIONAL_EDGE = {
    "stages": {"a": said(["conversational_query"], ["x"]), "b": said(["x"], ["y"])},
    "edges": [["START", "a", "b", "END"]],
    "conditional_edges": [
        asking("a", "is_query_mode", ["mode"], {True: "b", False: "END"})
    ],
}
TWO_CONDITIONAL_EDGES_ON_ONE_SOURCE = {
    "stages": {
        "a": said(["conversational_query"], ["x"]),
        "b": said(["x"], ["y"]),
        "c": said(["x"], ["z"]),
    },
    "edges": [["START", "a"], ["b", "END"], ["c", "END"]],
    "conditional_edges": [
        asking("a", "is_query_mode", ["mode"], {True: "b", False: "END"}),
        asking(
            "a", "has_history", ["conversational_history"], {True: "c", False: "END"}
        ),
    ],
}
SEVERAL_SOURCES = {
    "stages": {
        "a": said(["conversational_query"], ["x"]),
        "b": said(["x"], ["y"]),
        "c": said(["x"], ["z"]),
    },
    "edges": [["START", "a", "b", "END"], ["START", "c", "END"], ["a", "c"]],
}


@pytest.mark.parametrize(
    ("graph", "names", "have"),
    [
        (TWO_PLAIN_EDGES_OUT_OF_START, "'START'", "has"),
        (TWO_PLAIN_EDGES_OUT_OF_A_STAGE, "'a'", "has"),
        (A_PLAIN_AND_A_CONDITIONAL_EDGE, "'a'", "has"),
        (TWO_CONDITIONAL_EDGES_ON_ONE_SOURCE, "'a'", "has"),
        (SEVERAL_SOURCES, "'START', 'a'", "each have"),
    ],
    ids=[
        "two-plain-edges-out-of-start",
        "two-plain-edges-out-of-a-stage",
        "a-plain-and-a-conditional-edge",
        "two-conditional-edges-on-one-source",
        "several-sources",
    ],
)
def test_it_refuses_more_than_one_way_out_at_once_naming_the_stages(
    graph: dict[str, Any], names: str, have: str
) -> None:
    message = SEVERAL_WAYS_OUT.format(names=names, have=have)

    with pytest.raises(ValueError, match=re.escape(message)):
        PydanticGraphOrchestrator.check_graph_runs(described(graph))


def a_stage_named(name: str) -> dict[str, Any]:
    """A conditional edge at START to ``rewrite`` or to a stage named ``name``."""
    return {
        "stages": {
            "rewrite": said(["conversational_history"], ["standalone_query"]),
            name: said(["standalone_query"], ["answer"]),
        },
        "edges": [["rewrite", name, "END"]],
        "conditional_edges": [
            asking(
                "START",
                "has_history",
                ["conversational_history"],
                {True: "rewrite", False: name},
            )
        ],
    }


@pytest.mark.parametrize("name", ["START:has_history", "START:has_history:decision"])
def test_it_refuses_a_stage_named_as_a_node_of_a_conditional_edge(name: str) -> None:
    message = (
        f"stage {name!r} has the name of a node this orchestrator adds for the "
        "conditional edge out of 'START' (expected a stage name other than "
        "'START:has_history' and 'START:has_history:decision')"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        PydanticGraphOrchestrator.check_graph_runs(described(a_stage_named(name)))


def test_a_decision_is_a_node_named_after_its_conditional_edge() -> None:
    orchestrator = PydanticGraphOrchestrator(opened_of(described(ROUTED_SHAPE)))

    steps = set(orchestrator.graph.nodes)

    assert "route:label:decision" in steps
    assert not any(step.startswith("decision") for step in steps)


A_STAGE_WITH_NO_WAY_OUT = {
    "stages": {"a": said(["conversational_query"], ["x"]), "b": said(["x"], ["y"])},
    "edges": [["START", "a"]],
    "conditional_edges": [
        asking("a", "is_query_mode", ["mode"], {True: "b", False: "END"})
    ],
}
NOTHING_LEADS_TO_END = {
    "stages": {"a": said(["conversational_query"], ["x"])},
    "edges": [["START", "a"]],
}
A_STAGE_NO_EDGE_LEADS_TO = {
    "stages": {"a": said(["conversational_query"], ["x"]), "b": said(["x"], ["y"])},
    "edges": [["START", "a", "END"], ["b", "END"]],
}
# What the library says, then what it suggests (suppress=...) of its own API.
SUPPRESS = (
    " If this is intentional, you can suppress this error by passing "
    "`validate_graph_structure=False` to the call to `GraphBuilder.build`."
)


@pytest.mark.parametrize(
    ("graph", "said_by_the_library"),
    [
        (
            A_STAGE_WITH_NO_WAY_OUT,
            "The following nodes have no outgoing edges: ['b'].",
        ),
        (NOTHING_LEADS_TO_END, "The graph has no edges to the end node."),
        (
            A_STAGE_NO_EDGE_LEADS_TO,
            "The following nodes are not reachable from the start node: ['b'].",
        ),
    ],
    ids=["a-stage-with-no-way-out", "nothing-leads-to-end", "a-stage-no-edge-leads-to"],
)
def test_it_refuses_a_graph_its_library_does_not_build_in_the_library_s_words(
    graph: dict[str, Any], said_by_the_library: str
) -> None:
    message = (
        f"pydantic-graph does not build this graph: {said_by_the_library}{SUPPRESS} "
        "(expected every stage on a path from START to END)"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        PydanticGraphOrchestrator.check_graph_runs(described(graph))


A_LOOP_ON_ONE_STAGE = {
    "stages": {"draft": said(["conversational_query"], ["answer"])},
    "edges": [["START", "draft"]],
    "conditional_edges": [
        asking("draft", "is_query_mode", ["mode"], {True: "END", False: "draft"})
    ],
}
A_LOOP_THROUGH_TWO_STAGES = {
    "stages": {
        "draft": said(["conversational_query"], ["draft"]),
        "polish": said(["draft"], ["answer"]),
    },
    "edges": [["START", "draft", "polish"]],
    "conditional_edges": [
        asking("polish", "is_query_mode", ["mode"], {True: "END", False: "draft"})
    ],
}


@pytest.mark.parametrize(
    ("graph", "loop"),
    [
        (A_LOOP_ON_ONE_STAGE, "'draft' -> 'draft'"),
        (A_LOOP_THROUGH_TWO_STAGES, "'draft' -> 'polish' -> 'draft'"),
    ],
    ids=["on-one-stage", "through-two-stages"],
)
def test_it_refuses_a_loop_naming_its_stages(graph: dict[str, Any], loop: str) -> None:
    message = (
        f"the stages {loop} form a loop; pydantic-graph has no limit that stops "
        "one (expected a graph without a loop)"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        PydanticGraphOrchestrator.check_graph_runs(described(graph))
