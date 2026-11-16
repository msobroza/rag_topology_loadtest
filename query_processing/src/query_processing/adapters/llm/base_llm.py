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

from abc import abstractmethod

from ...domain.base_component import BaseComponent
from ...domain.component_requests import GenerateAnswer, GenerateRequest


class BaseLLM(BaseComponent[GenerateRequest, GenerateAnswer], component_type="llm"):
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

    async def ainvoke(self, request: GenerateRequest) -> GenerateAnswer:
        """Answer the request's conversation.

        Args:
            request: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            The model's answer text.

        Example:
            >>> await llm.ainvoke(GenerateRequest(
            ...     messages=[{"role": "user", "content": "Hi"}]))
            GenerateAnswer(text='Hello!')
        """
        return GenerateAnswer(text=await self.generate(request.messages))
