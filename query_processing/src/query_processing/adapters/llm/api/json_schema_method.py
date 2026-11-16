"""``json_schema``: the schema goes to the endpoint as OpenAI's ``response_format``.

The OVMS, vLLM and Ollama documentation say they accept that shape (spec 5).
It is the default of the LLM's ``structured_output`` setting.

Example:
    >>> JsonSchemaMethod().prepare(messages, Verdict).messages == messages
    True
"""

from __future__ import annotations

from pydantic import BaseModel

from .base_structured_output_method import (
    STRUCTURED_OUTPUT_METHODS,
    BaseStructuredOutputMethod,
    PreparedRequest,
)


@STRUCTURED_OUTPUT_METHODS.register("json_schema")
class JsonSchemaMethod(BaseStructuredOutputMethod):
    """Sends the schema as ``response_format`` and the messages unchanged.

    Example:
        >>> JsonSchemaMethod().json_text('  {"ok": true} ')
        '{"ok": true}'
    """

    def prepare(
        self, messages: list[dict[str, str]], output_schema: type[BaseModel]
    ) -> PreparedRequest:
        """Return the messages unchanged and the schema as ``response_format``.

        Args:
            messages: OpenAI-style messages.
            output_schema: The pydantic model the answer must follow.

        Returns:
            ``response_format`` is ``{"type": "json_schema", "json_schema":
            {name, schema, strict}}``, named after the model, strict.

        Example:
            >>> JsonSchemaMethod().prepare(messages, Verdict).response_format[
            ...     "json_schema"]["name"]
            'Verdict'
        """
        json_schema = {
            "name": output_schema.__name__,
            "schema": output_schema.model_json_schema(),
            "strict": True,
        }
        return PreparedRequest(
            messages, {"type": "json_schema", "json_schema": json_schema}
        )

    def json_text(self, answer: str) -> str:
        """Return the answer stripped: the endpoint imposed the schema.

        Args:
            answer: The answer text, perhaps with blank space around the object.

        Returns:
            The text stripped of surrounding blank space.

        Example:
            >>> JsonSchemaMethod().json_text('  {"ok": true} ')
            '{"ok": true}'
        """
        return answer.strip()
