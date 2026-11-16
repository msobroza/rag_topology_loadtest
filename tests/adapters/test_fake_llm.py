"""The ``fake`` LLM: an echo, with an optional latency."""

from __future__ import annotations

import time

from rag_load_test.adapters.llm.fake.fake_llm import FakeLLM


async def test_fake_llm_echoes_last_user_message_and_counts_calls() -> None:
    llm = FakeLLM()
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]

    assert (await llm.generate(messages)).text == "[fake-llm] second"
    answer = await llm.generate([{"role": "system", "content": "x"}])
    assert answer.text == "[fake-llm] "
    assert llm.calls == 2


async def test_fake_llm_honours_latency() -> None:
    llm = FakeLLM(latency_ms=30)
    start = time.perf_counter()
    await llm.generate([{"role": "user", "content": "hi"}])

    assert time.perf_counter() - start >= 0.025


async def test_the_fake_llm_counts_words_as_tokens() -> None:
    generation = await FakeLLM().generate(
        [{"role": "user", "content": "How many days"}]
    )
    assert (
        generation.text,
        generation.prompt_tokens,
        generation.completion_tokens,
    ) == (
        "[fake-llm] How many days",
        3,
        4,
    )
