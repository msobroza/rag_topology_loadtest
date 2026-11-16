"""The ``fake`` LLM: an echo, with an optional latency, or an instance of a schema."""

from __future__ import annotations

import datetime
import json
import time
from enum import StrEnum
from typing import Any, Literal

import pytest
from pydantic import ConfigDict, create_model

from query_processing.adapters.llm.fake.fake_llm import FakeLLM


async def test_fake_llm_echoes_last_user_message_and_counts_calls() -> None:
    llm = FakeLLM()
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]

    assert await llm.generate(messages) == "[fake-llm] second"
    assert await llm.generate([{"role": "system", "content": "x"}]) == "[fake-llm] "
    assert llm.calls == 2


# Built by call, not by class statement: no code reads these fields by name.
STRICT = ConfigDict(extra="forbid")
Kind = StrEnum("Kind", {"POLICY": "policy", "FORM": "form"})
Tone: Any = Literal["formal", "plain"]
Detail: Any = create_model(
    "Detail", __config__=STRICT, page=(int, ...), kind=(Kind, ...)
)
Rewritten = create_model(
    "Rewritten",
    __config__=STRICT,
    standalone_query=(str, ...),
    remark=(str, ...),
    conditions=(list[Detail], ...),
    detail=(Detail, ...),
    tone=(Tone, ...),
    sure=(bool, ...),
    score=(float, ...),
    tags=(dict[str, str], ...),
    note=(str | None, ...),
    lang=(str, "fr"),
)


async def test_under_a_schema_the_fake_llm_answers_an_instance_it_accepts() -> None:
    llm = FakeLLM()

    text = await llm.generate(
        [{"role": "user", "content": "and b?"}], output_schema=Rewritten
    )

    assert json.loads(text) == {
        "standalone_query": "[fake-llm] and b?",
        "remark": "",
        "conditions": [],
        "detail": {"page": 0, "kind": "policy"},
        "tone": "formal",
        "sure": False,
        "score": 0.0,
        "tags": {},
        "note": None,
        "lang": "fr",
    }
    assert Rewritten.model_validate_json(text).model_dump(mode="json") == json.loads(
        text
    )
    assert llm.calls == 1


async def test_the_fake_llm_refuses_a_schema_field_it_cannot_fill() -> None:
    odd = create_model("Odd", when=(datetime.date, ...))

    with pytest.raises(TypeError, match=r"'when'.*date.*expected"):
        await FakeLLM().generate([], output_schema=odd)


async def test_fake_llm_honours_latency() -> None:
    llm = FakeLLM(latency_ms=30)
    start = time.perf_counter()
    await llm.generate([{"role": "user", "content": "hi"}])

    assert time.perf_counter() - start >= 0.025
