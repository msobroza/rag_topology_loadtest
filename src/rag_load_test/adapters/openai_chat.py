"""The OpenAI chat model; ``openai`` is imported only when it is used."""

from __future__ import annotations

from typing import Any

from ..domain.errors import RagDependencyError


class OpenAIChatModel:
    """ChatModelPort on the OpenAI Chat Completions API (``AsyncOpenAI``).

    The SDK reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` itself, so a Domino AI
    Gateway can be substituted without code changes.

    Example::

        answer = await OpenAIChatModel.open("gpt-4o-mini").generate(messages)
    """

    def __init__(self, client: Any, model: str) -> None:
        self._client, self.model_name = client, model

    @classmethod
    def open(cls, model: str) -> OpenAIChatModel:
        import openai  # SDK defaults: 10 min timeout, 2 retries

        return cls(openai.AsyncOpenAI(), model)

    async def aclose(self) -> None:
        """Close the SDK's HTTP connections."""
        await self._client.close()

    async def generate(self, messages: list[dict[str, str]]) -> str:
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0
            )
        except openai.APITimeoutError as exc:
            raise RagDependencyError(
                "llm", str(exc), target=self.model_name, kind="timeout"
            ) from exc
        except openai.APIError as exc:
            raise RagDependencyError("llm", str(exc), target=self.model_name) from exc
        return response.choices[0].message.content or ""
