"""The graph description: stages, chains of plain edges and conditional edges."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_load_test.domain.graph_description import (
    ConditionalEdge,
    GraphDescription,
    StageDescription,
)

# The graph block of the spec's workflow file, as YAML loads it: true and false
# are booleans.
SPEC_GRAPH = {
    "stages": {
        "embed": {"call": "embedder"},
        "retrieve": {"call": "vector_store", "retrieval_mode": "dense", "top_k": 20},
        "rerank": {"call": "reranker", "top_k": 10},
        "generate": {"call": "llm", "system_prompt": "s", "user_prompt": "u"},
    },
    "edges": [["START", "embed", "retrieve", "rerank"], ["generate", "END"]],
    "conditional_edges": [
        {
            "source": "rerank",
            "path": "is_query_mode",
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


def test_a_stage_type_defaults_to_the_stage_name() -> None:
    assert StageDescription(call="embedder").type_name("embed") == "embed"
    assert StageDescription(type="embed", call="e").type_name("first") == "embed"


def test_every_key_besides_type_and_call_is_a_setting_of_the_stage() -> None:
    graph = GraphDescription.model_validate(SPEC_GRAPH)

    assert graph.stages["retrieve"].call == "vector_store"
    assert graph.stages["retrieve"].stage_settings == {
        "retrieval_mode": "dense",
        "top_k": 20,
    }
    assert graph.stages["embed"].stage_settings == {}


def test_a_path_map_goes_from_true_and_false_written_as_text_too() -> None:
    # As an environment variable would give it.
    path_map = {"true": "generate", "false": "END"}

    edge = ConditionalEdge(source="rerank", path="is_query_mode", path_map=path_map)

    assert edge.path_map == {True: "generate", False: "END"}


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
