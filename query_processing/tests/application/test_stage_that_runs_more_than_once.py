"""A stage that runs more than once in a request reports the sum of its runs.

Two graphs ``check_graph`` accepts run one stage twice: two branches that join
at different depths, and a loop on one stage. ``timings_ms`` and ``tokens``
hold every run, and a stage that ran once reports that run alone.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.application.edge_conditions.edge_condition import (
    EDGE_CONDITIONS,
    EdgeCondition,
)
from query_processing.application.graph_check import check_graph
from query_processing.application.stages.base_stage import StageContext
from query_processing.application.workflow import RagResult
from query_processing.application.workflow_builder import build_workflow
from query_processing.domain.component_requests import GenerateAnswer
from query_processing.domain.graph_description import GraphDescription
from query_processing.domain.prompts import PromptMessage
from query_processing.domain.registry import Registry

# What each call of the LLM costs, and how long it takes.
PROMPT_TOKENS, COMPLETION_TOKENS, CALL_MS = 10, 1, 50.0
CONTEXT = StageContext({"say": [PromptMessage(role="user", content="say")]})


class SameCostLLM(FakeLLM):
    """An LLM whose every call costs the same tokens and time; answers ``call n``."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        self.calls += 1
        await asyncio.sleep(CALL_MS / 1000.0)
        return GenerateAnswer(
            text=f"call {self.calls}",
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
        )


def asked(reads: list[str], writes: list[str]) -> dict[str, Any]:
    """A generate stage of the LLM, which reads and writes these fields."""
    return {
        "type": "generate",
        "call": "llm",
        "prompt": "say",
        "reads": reads,
        "writes": writes,
    }


# No loop: two branches leave START and join at 'c' at different depths, so 'c'
# runs once after 'a' and once after 'd'.
JOINED_AT_TWO_DEPTHS = {
    "stages": {
        "a": asked(["conversational_query"], ["draft"]),
        "b": asked(["conversational_query"], ["note"]),
        "d": asked(["note"], ["draft"]),
        "c": asked(["draft"], ["answer"]),
    },
    "edges": [["START", "a", "c", "END"], ["START", "b", "d", "c"]],
}
# A loop: 'draft' is asked again until its second answer.
LOOP_ON_ONE_STAGE = {
    "stages": {"draft": asked(["conversational_query"], ["answer"])},
    "edges": [["START", "draft"]],
    "conditional_edges": [
        {
            "source": "draft",
            "path": "is_second_call",
            "reads": ["answer"],
            "path_map": {True: "END", False: "draft"},
        }
    ],
}


def is_second_call(answer: str) -> bool:
    return answer == "call 2"


LOOP_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")
LOOP_CONDITIONS.register("is_second_call")(is_second_call)


async def run_graph(
    graph: dict[str, Any], conditions: Registry[EdgeCondition] = EDGE_CONDITIONS
) -> tuple[RagResult, SameCostLLM]:
    """Check and run ``graph`` over a counting LLM; the result and the LLM."""
    description = GraphDescription.model_validate(graph)
    check_graph(description, {"llm": "llm"}, CONTEXT, edge_conditions=conditions)
    llm = SameCostLLM()
    workflow = build_workflow(
        description, {"llm": llm}, CONTEXT, edge_conditions=conditions
    )
    return await workflow.run("q"), llm


@pytest.mark.parametrize(
    ("graph", "conditions", "runs"),
    [
        (JOINED_AT_TWO_DEPTHS, EDGE_CONDITIONS, {"a": 1, "b": 1, "d": 1, "c": 2}),
        (LOOP_ON_ONE_STAGE, LOOP_CONDITIONS, {"draft": 2}),
    ],
    ids=["joined-at-two-depths", "loop"],
)
async def test_every_run_of_a_stage_is_reported(
    graph: dict[str, Any],
    conditions: Registry[EdgeCondition],
    runs: dict[str, int],
) -> None:
    result, llm = await run_graph(graph, conditions)

    assert llm.calls == sum(runs.values())
    assert result.tokens == {
        stage: {"prompt": n * PROMPT_TOKENS, "completion": n * COMPLETION_TOKENS}
        for stage, n in runs.items()
    }
    for stage, n in runs.items():
        # Each run sleeps CALL_MS, so the stage's time holds all of them. No
        # upper bound: a loaded machine adds what it likes to a sleep, and a
        # run counted twice shows in the tokens above, which are exact.
        assert result.timings_ms[stage] >= n * CALL_MS


async def test_a_stage_that_runs_once_reports_that_run_alone() -> None:
    graph = {"stages": {"draft": asked(["conversational_query"], ["answer"])}}
    result, _ = await run_graph({**graph, "edges": [["START", "draft", "END"]]})

    assert result.tokens == {
        "draft": {"prompt": PROMPT_TOKENS, "completion": COMPLETION_TOKENS}
    }
    # The exact tokens above show the run was counted once; its time has no
    # upper bound a loaded machine could not pass.
    assert result.timings_ms["draft"] >= CALL_MS
