"""The LLM port: answers an OpenAI-style message list.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

Example:
    >>> class MyLLM(BaseLLM):
    ...     async def generate(self, messages, output_schema=None):
    ...         return GenerateAnswer(text="...")
"""

from __future__ import annotations

from abc import abstractmethod

from pydantic import BaseModel

from ...domain.base_component import BaseComponent
from ...domain.component_requests import GenerateAnswer, GenerateRequest


class BaseLLM(BaseComponent[GenerateRequest, GenerateAnswer], component_type="llm"):
    """Answers an OpenAI-style message list."""

    model_name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
    ) -> GenerateAnswer:
        """Answer a conversation, as text or as one instance of a schema.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            output_schema: The pydantic model the answer must follow; None for
                free text.

        Returns:
            The model's answer text, under ``output_schema`` the JSON object
            alone, which the schema accepts; with the token counts the model
            reported, each None when it reported none.

        Example:
            >>> await llm.generate(messages, output_schema=Verdict)
            GenerateAnswer(text='{"ok": true}', prompt_tokens=31, completion_tokens=4)
        """

    async def ainvoke(self, request: GenerateRequest) -> GenerateAnswer:
        """Answer the request's conversation, under its output schema if it has one.

        Args:
            request: OpenAI-style messages, each with ``role`` and ``content``,
                and the optional schema the answer must follow.

        Returns:
            What ``generate`` answered: the text and its token counts.

        Example:
            >>> await llm.ainvoke(GenerateRequest(
            ...     messages=[{"role": "user", "content": "Hi"}]))
            GenerateAnswer(text='Hello!', prompt_tokens=9, completion_tokens=2)
        """
        return await self.generate(request.messages, request.output_schema)
