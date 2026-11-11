"""The OpenAI chat model; ``openai`` is imported only when it is used."""

from __future__ import annotations

from typing import Any

from ...domain.errors import RagDependencyError
from ...domain.ports import ChatModelPort


class OpenAIChatModel(ChatModelPort):
    """ChatModelPort on the OpenAI Chat Completions API (``AsyncOpenAI``).

    The SDK reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` itself, so a Domino AI
    Gateway can be substituted without code changes.

    Args:
        client: An ``openai.AsyncOpenAI`` client, or a stand-in with the same
            ``chat.completions.create`` and ``close``.
        model: The chat model name sent with every request.
        temperature: The sampling temperature sent with every request.

    Example:
        >>> chat = OpenAIChatModel.open("gpt-4o-mini", temperature=0.0)
        >>> answer = await chat.generate(messages)
    """

    def __init__(self, client: Any, model: str, *, temperature: float) -> None:
        self._client, self.model_name = client, model
        self.temperature = temperature

    @classmethod
    def open(cls, model: str, *, temperature: float) -> OpenAIChatModel:
        """Build the model on a fresh ``AsyncOpenAI`` client, importing ``openai``.

        Args:
            model: The chat model name sent with every request.
            temperature: The sampling temperature sent with every request.

        Returns:
            The chat model, with the SDK's defaults (10 min timeout, 2 retries).
        """
        import openai  # SDK defaults: 10 min timeout, 2 retries

        return cls(openai.AsyncOpenAI(), model, temperature=temperature)

    async def aclose(self) -> None:
        """Close the SDK's HTTP connections."""
        await self._client.close()

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Answer a conversation at the configured temperature.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            The model's answer text; "" when the API returns no content.

        Raises:
            RagDependencyError: If the API call times out (kind ``timeout``) or
                fails with any other ``openai.APIError``.
        """
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
        except openai.APITimeoutError as exc:
            raise RagDependencyError(
                "llm", str(exc), target=self.model_name, kind="timeout"
            ) from exc
        except openai.APIError as exc:
            raise RagDependencyError("llm", str(exc), target=self.model_name) from exc
        return response.choices[0].message.content or ""
