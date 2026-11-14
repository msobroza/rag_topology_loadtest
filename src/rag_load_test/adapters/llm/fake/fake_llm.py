"""The ``fake`` LLM: echoes the last user message, after an optional latency.

Example:
    >>> await FakeLLM().generate([{"role": "user", "content": "Hi"}])
    '[fake-llm] Hi'
"""

from __future__ import annotations

import asyncio

from ..base_llm import BaseLLM


class FakeLLM(BaseLLM):
    """Echoes the last user message; ``latency_ms`` imitates generation time.

    Args:
        latency_ms: Milliseconds each ``generate`` sleeps first; 0 answers at once.
    """

    model_name = "fake-llm"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms, self.calls = latency_ms, 0

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Echo the last user message, after ``latency_ms``; counts the calls.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            ``"[fake-llm] "`` and the first 200 characters of the last user message
            ("" when there is none).
        """
        self.calls += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"[fake-llm] {user[:200]}"
