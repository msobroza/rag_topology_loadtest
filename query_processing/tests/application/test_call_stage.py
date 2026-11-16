"""A plain call: its keys and reads in as the request, the answer out as its writes."""

from __future__ import annotations

from echo_components import BaseEcho, EchoRequest, Upper
from query_processing.application.stage_binding import bind_call
from query_processing.application.stages.base_stage import StageFields
from query_processing.application.stages.call_stage import CallStage


async def test_a_plain_call_hands_its_reads_and_writes_the_answer() -> None:
    stage = CallStage(
        Upper(), bind_call(BaseEcho, {}, StageFields(("message",), ("shout", "size")))
    )

    assert await stage.run({"message": "ab"}) == {"shout": "AB", "size": 2}


async def test_a_key_fills_its_attribute_and_the_state_is_not_read_for_it() -> None:
    echo = Upper()
    stage = CallStage(
        echo, bind_call(BaseEcho, {"text": "hi"}, StageFields((), ("shout",)))
    )

    assert await stage.run({}) == {"shout": "HI"}
    assert echo.asked == [EchoRequest(text="hi")]
