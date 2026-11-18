"""A generate stage caps its answer: its ``max_tokens`` reaches the LLM.

The cap is the stage's, not the LLM's: two stages that call one LLM, a rewrite
under an output schema and a free answer, may cap their answers apart.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import ValidationError

from query_processing.application.graph_check import WorkflowGraphError
from query_processing.application.stages.base_stage import STAGE_TYPES, StageContext
from query_processing.application.stages.generate_stage import GenerateSettings
from query_processing.application.workflow_builder import open_stage
from query_processing.domain.graph_description import StageDescription
from query_processing.domain.prompts import PromptMessage
from workflow_fakes import CapRecordingLLM

MESSAGES = [{"role": "user", "content": "How many days off?"}]
CONTEXT = StageContext(
    {"answer": [PromptMessage(role="user", content="{{ conversational_query }}")]}
)


async def run_stage(llm: CapRecordingLLM, stage: dict[str, Any]) -> None:
    """Open ``stage``, calling ``llm`` as ``llm``, and run it once."""
    description = StageDescription.model_validate(stage)
    opened = open_stage("answer", description, {"llm": llm}, STAGE_TYPES, CONTEXT)
    question = MESSAGES[0]["content"]
    await opened.run({"conversational_query": question, "conversation": MESSAGES})


GENERATE_STAGE = {
    "type": "generate",
    "call": "llm",
    "prompt": "answer",
    "reads": ["conversational_query"],
    "writes": ["answer"],
}
# A plain call fills every attribute of the request: its keys the schema (none
# here) and the cap, its reads the messages.
PLAIN_CALL = {
    "call": "llm",
    "output_schema": None,
    "reads": ["conversation"],
    "writes": ["answer"],
}


@pytest.mark.parametrize(
    "stage", [GENERATE_STAGE, PLAIN_CALL], ids=["generate-stage", "plain-call"]
)
async def test_a_generate_stage_and_a_plain_call_send_the_same_cap(
    stage: dict[str, Any],
) -> None:
    llm = CapRecordingLLM()

    await run_stage(llm, {**stage, "max_tokens": 64})

    assert llm.caps == [64]


async def test_a_generate_stage_that_states_no_cap_sends_none() -> None:
    llm = CapRecordingLLM()

    await run_stage(llm, GENERATE_STAGE)

    assert GenerateSettings(prompt="answer").max_tokens is None
    assert llm.caps == [None]


@pytest.mark.parametrize("cap", [0, -3])
def test_a_cap_below_one_is_refused_naming_the_value_and_the_shape(cap: int) -> None:
    with pytest.raises(ValidationError) as refused:
        GenerateSettings(prompt="answer", max_tokens=cap)

    [error] = refused.value.errors()
    assert error["loc"] == ("max_tokens",)
    assert f"max_tokens is {cap} (expected a whole number of at least 1" in str(
        error["msg"]
    )


@pytest.mark.parametrize(
    "stage", [GENERATE_STAGE, PLAIN_CALL], ids=["generate-stage", "plain-call"]
)
def test_a_generate_stage_and_a_plain_call_refuse_a_cap_of_zero_when_opened(
    stage: dict[str, Any],
) -> None:
    description = StageDescription.model_validate({**stage, "max_tokens": 0})

    with pytest.raises(WorkflowGraphError) as refused:
        open_stage(
            "answer", description, {"llm": CapRecordingLLM()}, STAGE_TYPES, CONTEXT
        )

    said = str(refused.value)
    assert re.search(r"^stage 'answer': .*max_tokens'? is 0 \(expected", said, re.S)
