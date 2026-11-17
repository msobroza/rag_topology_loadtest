"""A plain call: its keys and reads in as the request, the answer out as its writes."""

from __future__ import annotations

import pytest

from echo_components import BaseEcho, EchoRequest, Upper
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.router.base_router import BaseRouter
from query_processing.adapters.router.fake.fake_router import FakeRouter
from query_processing.application.stage_binding import bind_call
from query_processing.application.stages.base_stage import StageFields
from query_processing.application.stages.call_stage import CallStage
from query_processing.domain.errors import RagDependencyError

# A plain call fills every attribute of its request: an LLM's schema and cap
# are keys here, stated as none, and its reads fill the messages.
UNCAPPED_TEXT = {"output_schema": None, "max_tokens": None}


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


async def test_a_plain_call_on_an_llm_writes_its_token_counts() -> None:
    fields = StageFields(("messages",), ("reply", "read", "written"))
    stage = CallStage(FakeLLM(), bind_call(BaseLLM, UNCAPPED_TEXT, fields))

    written = await stage.run({"messages": [{"role": "user", "content": "how many"}]})

    assert written == {"reply": "[fake-llm] how many", "read": 2, "written": 3}


async def test_a_plain_call_on_an_llm_can_write_its_text_alone() -> None:
    fields = StageFields(("messages",), ("reply",))
    stage = CallStage(FakeLLM(), bind_call(BaseLLM, UNCAPPED_TEXT, fields))

    written = await stage.run({"messages": [{"role": "user", "content": "hi"}]})

    assert written == {"reply": "[fake-llm] hi"}


def routing_stage(router: FakeRouter) -> CallStage:
    """A plain call on a router whose routes are a and b, writing ``route``."""
    fields = StageFields(("routing_embedding",), ("route",))
    return CallStage(router, bind_call(BaseRouter, {}, fields, {"name": ("a", "b")}))


async def test_a_label_among_those_its_field_declares_is_written() -> None:
    stage = routing_stage(FakeRouter(route="b"))

    assert await stage.run({"routing_embedding": [0.1]}) == {"route": "b"}


async def test_an_answer_outside_its_fields_labels_is_a_dependency_error() -> None:
    stage = routing_stage(FakeRouter(route="zzz"))

    with pytest.raises(RagDependencyError) as failed:
        await stage.run({"routing_embedding": [0.1]})

    assert str(failed.value) == (
        "router unavailable: answered 'zzz' for 'route' (expected one of the "
        "labels of 'route': 'a', 'b')"
    )
