"""The LLM port: answers an OpenAI-style message list.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

Example:
    >>> class MyLLM(BaseLLM):
    ...     async def generate(self, messages): return "..."
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Answers an OpenAI-style message list."""

    model_name: str

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Answer a conversation.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            The model's answer text.
        """
