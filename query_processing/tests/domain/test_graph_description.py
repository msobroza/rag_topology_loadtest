"""The graph description: stages, chains of plain edges and conditional edges."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from query_processing.domain.graph_description import (
    ConditionalEdge,
    GraphDescription,
    StageDescription,
)

# The graph block of the spec's workflow file, as YAML loads it: true and false
# are booleans.
SPEC_GRAPH = {
    "stages": {
        "embed": {
            "call": "embedder",
            "reads": ["standalone_query"],
            "writes": ["query_embedding"],
        },
        "retrieve": {
            "call": "vector_store",
            "retrieval_mode": "dense",
            "top_k": 20,
            "reads": ["query_embedding"],
            "writes": ["candidates"],
        },
        "rerank": {
            "call": "reranker",
            "top_k": 10,
            "reads": ["standalone_query", "candidates"],
            "writes": ["reranked"],
        },
        "generate": {
            "call": "llm",
            "prompt": "answer_from_context",
            "reads": ["conversational_query", "reranked"],
            "writes": ["answer"],
        },
    },
    "edges": [["START", "embed", "retrieve", "rerank"], ["generate", "END"]],
    "conditional_edges": [
        {
            "source": "rerank",
            "path": "is_query_mode",
            "reads": ["mode"],
            "path_map": {True: "generate", False: "END"},
        }
    ],
}


def test_each_chain_is_one_edge_per_pair_of_neighbours() -> None:
    graph = GraphDescription.model_validate(SPEC_GRAPH)

    assert graph.edge_pairs() == [
        ("START", "embed"),
        ("embed", "retrieve"),
        ("retrieve", "rerank"),
        ("generate", "END"),
    ]


def test_the_links_are_the_plain_edges_then_each_conditional_edge_row() -> None:
    graph = GraphDescription.model_validate(
        {
            "stages": {},
            "edges": [["START", "embed", "rerank"]],
            "conditional_edges": [
                {
                    "source": "rerank",
                    "path": "is_query_mode",
                    "reads": ["mode"],
                    "path_map": {True: "generate", False: "END"},
                }
            ],
        }
    )

    assert graph.links() == [
        ("START", "embed"),
        ("embed", "rerank"),
        ("rerank", "generate"),
        ("rerank", "END"),
    ]


def test_a_graph_without_conditional_edges_links_its_plain_edges_alone() -> None:
    graph = GraphDescription(stages={}, edges=[["START", "embed", "END"]])

    assert graph.links() == [("START", "embed"), ("embed", "END")]


def test_a_stage_type_defaults_to_the_stage_name() -> None:
    lists = {"reads": ["a"], "writes": ["b"]}

    assert StageDescription(call="llm", **lists).type_name("generate") == "generate"
    assert StageDescription(type="generate", call="e", **lists).type_name("first") == (
        "generate"
    )


@pytest.mark.parametrize("missing", ["reads", "writes"])
def test_a_stage_states_both_its_lists(missing: str) -> None:
    stages = {name: dict(stage) for name, stage in SPEC_GRAPH["stages"].items()}
    del stages["rerank"][missing]

    with pytest.raises(ValidationError) as refused:
        GraphDescription.model_validate({**SPEC_GRAPH, "stages": stages})

    [error] = refused.value.errors()
    assert (error["loc"], error["type"]) == (("stages", "rerank", missing), "missing")


def test_every_key_besides_type_call_and_the_lists_is_a_setting_of_the_stage() -> None:
    graph = GraphDescription.model_validate(SPEC_GRAPH)

    assert graph.stages["retrieve"].call == "vector_store"
    assert graph.stages["retrieve"].stage_settings == {
        "retrieval_mode": "dense",
        "top_k": 20,
    }
    assert graph.stages["embed"].stage_settings == {}


def test_a_path_map_goes_from_true_and_false_written_as_text_too() -> None:
    # As an environment variable would give it.
    path_map = {"true": "generate", "False": "END"}

    edge = ConditionalEdge(
        source="rerank", path="is_query_mode", reads=["mode"], path_map=path_map
    )

    assert edge.path_map == {True: "generate", False: "END"}


def test_a_path_map_goes_from_labels_too() -> None:
    path_map = {"self_contained": "embed", "leans_on_history": "rewrite"}

    edge = ConditionalEdge(
        source="route", path="label", reads=["route"], path_map=path_map
    )

    assert edge.path_map == path_map


def test_a_conditional_edge_states_what_it_reads() -> None:
    unstated = {
        k: v for k, v in SPEC_GRAPH["conditional_edges"][0].items() if k != "reads"
    }

    with pytest.raises(ValidationError) as refused:
        GraphDescription.model_validate({**SPEC_GRAPH, "conditional_edges": [unstated]})

    [error] = refused.value.errors()
    assert (error["loc"], error["type"]) == (
        ("conditional_edges", 0, "reads"),
        "missing",
    )


def test_a_chain_of_one_name_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 2 items"):
        GraphDescription.model_validate({"stages": {}, "edges": [["START"]]})


@pytest.mark.parametrize(
    "graph",
    [
        {**SPEC_GRAPH, "nodes": {}},  # LangGraph's word, not the file's
        {
            **SPEC_GRAPH,
            "conditional_edges": [{**SPEC_GRAPH["conditional_edges"][0], "then": 1}],
        },
    ],
)
def test_an_unknown_key_of_the_graph_or_of_an_edge_is_refused(graph: dict) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GraphDescription.model_validate(graph)
