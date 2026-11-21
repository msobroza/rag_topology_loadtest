"""The LangGraph orchestrator: its state, its reserved names, its rows, and its
check, which asks LangGraph which stage names it takes.

The runs of whole graphs on it are the tests of ``application/``: the
workflow, the builder, and a stage that runs more than once.
"""

from __future__ import annotations

import re
from typing import Any, get_type_hints

import pytest
from langgraph.graph import END, START, StateGraph
from query_processing_core.retrieval.metadata_filter import MetadataCondition

from query_processing.adapters.orchestrator.langgraph.langgraph_orchestrator import (
    LangGraphOrchestrator,
    langgraph_name,
    path_map,
    state_type,
)
from query_processing.application.state_fields import (
    STATE_REDUCERS,
    field_types_of,
    state_field_types,
)
from query_processing.application.workflow_state import (
    METADATA_FILTER_FIELD,
    initial_state,
    sum_timings_by_stage,
    sum_tokens_by_stage,
)
from query_processing.domain.graph_description import ConditionalEdge, GraphDescription
from rag_graphs import fixture_graph, rag_graph

ON_PAY = MetadataCondition(attribute="topic", comparator="eq", value="pay")
ON_LEAVE = MetadataCondition(attribute="topic", comparator="ne", value="leave")
# The conditional edge of the builder's fixture graph.
DRAFT_EDGE = {
    "source": "draft",
    "path": "is_a_question",
    "reads": ["conversational_query"],
    "path_map": {True: "shout", False: "END"},
}


def test_the_state_type_holds_the_fields_the_timings_and_the_tokens() -> None:
    state = state_type(state_field_types({"draft": str}), STATE_REDUCERS)
    hints = get_type_hints(state, include_extras=True)

    assert hints["draft"] is str
    assert hints["timings_ms"].__metadata__ == (sum_timings_by_stage,)
    assert hints["tokens"].__metadata__ == (sum_tokens_by_stage,)


async def test_a_stage_that_writes_the_metadata_filter_adds_conditions() -> None:
    state = state_type(state_field_types(field_types_of({})), STATE_REDUCERS)
    graph = StateGraph(state)
    graph.add_node("narrow", lambda state: {METADATA_FILTER_FIELD: [ON_LEAVE]})
    graph.add_edge(START, "narrow")
    graph.add_edge("narrow", END)

    final = await graph.compile().ainvoke(
        initial_state("q?", "retrieve", metadata_filter=[ON_PAY])
    )

    assert final[METADATA_FILTER_FIELD] == [ON_PAY, ON_LEAVE]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("START", "__start__"), ("END", "__end__"), ("draft", "draft")],
)
def test_langgraph_name_spells_the_reserved_names_langgraph_s_way(
    name: str, expected: str
) -> None:
    assert langgraph_name(name) == expected


def test_path_map_leads_each_answer_where_langgraph_knows_the_target() -> None:
    edge = ConditionalEdge.model_validate(DRAFT_EDGE)

    assert path_map(edge) == {True: "shout", False: "__end__"}


def said(reads: list[str], writes: list[str]) -> dict[str, Any]:
    """A generate stage of the LLM, which reads and writes these fields."""
    return {
        "type": "generate",
        "call": "llm",
        "prompt": "say",
        "reads": reads,
        "writes": writes,
    }


def is_query_mode_edge(source: str, path_map: dict[bool, str]) -> dict[str, Any]:
    """A conditional edge out of ``source`` that asks whether the mode is query."""
    edge = {"source": source, "path": "is_query_mode", "reads": ["mode"]}
    return {**edge, "path_map": path_map}


# Shapes a library may not run, besides the RAG's: two ways out of START that
# join at two depths, and a loop entered by a conditional edge at START.
JOINED_AT_TWO_DEPTHS = {
    "stages": {
        "a": said(["conversational_query"], ["draft"]),
        "b": said(["conversational_query"], ["note"]),
        "d": said(["note"], ["draft"]),
        "c": said(["draft"], ["answer"]),
    },
    "edges": [["START", "a", "c", "END"], ["START", "b", "d", "c"]],
}
LOOP_ENTERED_AT_START = {
    "stages": {"draft": said(["conversational_query"], ["answer"])},
    "conditional_edges": [
        is_query_mode_edge("START", {True: "draft", False: "END"}),
        is_query_mode_edge("draft", {True: "END", False: "draft"}),
    ],
}


@pytest.mark.parametrize(
    "graph",
    [
        fixture_graph(),
        *(rag_graph(mode).model_dump() for mode in ("dense", "lexical", "hybrid")),
        JOINED_AT_TWO_DEPTHS,
        LOOP_ENTERED_AT_START,
    ],
    ids=["fixture", "dense", "lexical", "hybrid", "joined-at-two-depths", "loop"],
)
def test_langgraph_accepts_the_graphs_start_up_accepts(graph: dict[str, Any]) -> None:
    LangGraphOrchestrator.check_graph_runs(GraphDescription.model_validate(graph))


@pytest.mark.parametrize("character", [":", "|"])
def test_it_refuses_a_stage_name_langgraph_does_not_take_with_langgraph_s_reason(
    character: str,
) -> None:
    name = f"re{character}rank"
    graph = {
        "stages": {name: said(["conversational_query"], ["answer"])},
        "edges": [["START", name, "END"]],
    }
    message = (
        f"stage {name!r} is no name LangGraph takes for a node: {character!r} is a "
        "reserved character and is not allowed in the node names. (expected a "
        "stage name LangGraph takes)"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LangGraphOrchestrator.check_graph_runs(GraphDescription.model_validate(graph))
