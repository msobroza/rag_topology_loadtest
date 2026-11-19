"""The LangGraph orchestrator: its state, its reserved names and its rows.

The runs of whole graphs on it are the tests of ``application/``: the
workflow, the builder, and a stage that runs more than once.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest
from langgraph.graph import END, START, StateGraph

from query_processing.adapters.orchestrator.langgraph.langgraph_orchestrator import (
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
from query_processing.domain.graph_description import ConditionalEdge
from query_processing.domain.retrieval.metadata_filter import MetadataCondition

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
