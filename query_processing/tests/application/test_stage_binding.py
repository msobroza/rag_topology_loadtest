"""How a stage's keys, ``reads`` and ``writes`` bind it to its component (spec 3.1)."""

from __future__ import annotations

from typing import Any

import pytest

from echo_components import BaseEcho
from query_processing.adapters.reranker.base_reranker import BaseReranker
from query_processing.application.stage_binding import bind_call, stage_type_of
from query_processing.application.stages.base_stage import STAGE_TYPES, StageFields
from query_processing.domain.graph_description import StageDescription
from query_processing.domain.passages import ScoredPassage


def test_keys_fill_named_attributes_and_reads_fill_the_rest_in_order() -> None:
    binding = bind_call(
        BaseReranker,
        {"top_k": 10},
        StageFields(("standalone_query", "candidates"), ("reranked",)),
    )

    assert binding.from_keys == {"top_k": 10}
    assert binding.from_reads == {"query": "standalone_query", "passages": "candidates"}
    assert binding.to_writes == {"reranked": "passages"}
    assert binding.field_types.reads == {
        "standalone_query": str,
        "candidates": list[ScoredPassage],
    }
    assert binding.field_types.writes == {"reranked": list[ScoredPassage]}


def test_writes_can_stop_early() -> None:
    binding = bind_call(BaseEcho, {}, StageFields(("message",), ("shout",)))

    assert binding.to_writes == {"shout": "text"}


def test_a_wrong_count_shows_what_the_component_takes_and_gives() -> None:
    with pytest.raises(ValueError, match="reads lists") as refused:
        bind_call(BaseEcho, {}, StageFields(("a", "b"), ("shout",)))

    assert str(refused.value) == (
        "echo takes (text: str) and gives (text: str, length: int); "
        "reads lists 2 fields"
    )


@pytest.mark.parametrize(
    ("keys", "fields", "problem"),
    [
        ({}, StageFields((), ("shout",)), "reads lists 0 fields"),
        (
            {"text": "hi"},
            StageFields(("a",), ()),
            "reads lists 1 field after its keys fill text",
        ),
        ({}, StageFields(("a",), ("b", "c", "d")), "writes lists 3 fields"),
        ({}, StageFields(("a",), ("b", "b")), "writes lists a field twice"),
        ({"loud": True}, StageFields(("a",), ()), "key 'loud' names no attribute"),
    ],
)
def test_lists_and_keys_that_do_not_fill_the_request_once_are_refused(
    keys: dict[str, Any], fields: StageFields, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        bind_call(BaseEcho, keys, fields)


def test_a_stage_is_of_its_type_else_of_its_name_else_a_plain_call() -> None:
    plain = StageDescription.model_validate(
        {"call": "embedder", "reads": ["a"], "writes": ["b"]}
    )
    typed = StageDescription.model_validate(
        {"type": "retrieve", "call": "vector_store", "reads": ["a"], "writes": ["b"]}
    )

    assert stage_type_of("embed", plain, STAGE_TYPES) is None
    assert stage_type_of("retrieve", plain, STAGE_TYPES) is STAGE_TYPES.get("retrieve")
    assert stage_type_of("first", typed, STAGE_TYPES) is STAGE_TYPES.get("retrieve")


def test_a_type_that_names_no_stage_type_is_refused_naming_the_known_ones() -> None:
    misnamed = StageDescription.model_validate(
        {"type": "rerank", "call": "reranker", "reads": ["a"], "writes": ["b"]}
    )

    with pytest.raises(ValueError, match=r"'rerank' \(expected one of: 'generate'"):
        stage_type_of("rerank", misnamed, STAGE_TYPES)


def test_the_lists_are_no_setting_of_the_stage() -> None:
    stage = StageDescription.model_validate(
        {"call": "reranker", "top_k": 10, "reads": ["a"], "writes": ["b"]}
    )

    assert stage.stage_settings == {"top_k": 10}
