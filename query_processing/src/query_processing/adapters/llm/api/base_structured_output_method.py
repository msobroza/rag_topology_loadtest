"""How an output schema reaches an endpoint, and the registry of the ways by name.

A structured output method prepares a request under a schema (the messages
sent and OpenAI's ``response_format``, or none) and reads the JSON object out
of the answer. The LLM's ``structured_output`` setting, in its ``api`` and
``served`` blocks, names one (spec 5); the ``api`` adapter applies it.

A new method is one new module beside this one, its class decorated with
``@STRUCTURED_OUTPUT_METHODS.register(name)``; the package imports every module.

Example:
    >>> STRUCTURED_OUTPUT_METHODS.get("json_schema")
    <class 'JsonSchemaMethod'>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

from pydantic import BaseModel

from ....domain.registry import Registry


class PreparedRequest(NamedTuple):
    """What is sent under a schema: the messages and OpenAI's ``response_format``.

    Example:
        >>> PreparedRequest(messages, None).response_format is None
        True
    """

    messages: list[dict[str, str]]
    response_format: dict[str, object] | None


class BaseStructuredOutputMethod(ABC):
    """One way of asking an endpoint for an answer that follows a schema.

    Example:
        >>> method = STRUCTURED_OUTPUT_METHODS.get("json_schema")()
        >>> method.prepare(messages, Verdict).response_format["type"]
        'json_schema'
    """

    @abstractmethod
    def prepare(
        self, messages: list[dict[str, str]], output_schema: type[BaseModel]
    ) -> PreparedRequest:
        """Return what to send so that the answer follows ``output_schema``.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``;
                left unchanged.
            output_schema: The pydantic model the answer must follow.

        Returns:
            The messages to send and the ``response_format``, or None.
        """

    @abstractmethod
    def json_text(self, answer: str) -> str:
        """Return the JSON object an answer holds, as text.

        Args:
            answer: The answer text.

        Returns:
            The object's text; the answer as it came, stripped, when it holds
            none, so that the schema refuses it and the answer is asked again.
        """


# Every structured output method by name; an LLM's structured_output names one.
STRUCTURED_OUTPUT_METHODS: Registry[type[BaseStructuredOutputMethod]] = Registry(
    "structured output method"
)
