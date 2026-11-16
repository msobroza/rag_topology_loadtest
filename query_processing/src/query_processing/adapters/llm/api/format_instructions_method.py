"""``format_instructions``: the schema is written in the messages.

It is for an endpoint that imposes no schema. The instruction and the parser
are LangChain core's (spec 5); no class of the library leaves this module.

Example:
    >>> FormatInstructionsMethod().prepare(messages, Verdict).response_format is None
    True
"""

from __future__ import annotations

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel

from .base_structured_output_method import (
    STRUCTURED_OUTPUT_METHODS,
    BaseStructuredOutputMethod,
    PreparedRequest,
)


@STRUCTURED_OUTPUT_METHODS.register("format_instructions")
class FormatInstructionsMethod(BaseStructuredOutputMethod):
    r"""Appends the schema as a ``user`` message and reads the object of a code fence.

    Example:
        >>> FormatInstructionsMethod().json_text('```json\n{"ok": true}\n```')
        '{"ok": true}'
    """

    def prepare(
        self, messages: list[dict[str, str]], output_schema: type[BaseModel]
    ) -> PreparedRequest:
        """Return the messages with the schema's instruction last.

        Args:
            messages: OpenAI-style messages; the list given is not changed.
            output_schema: The pydantic model the answer must follow.

        Returns:
            The messages and one ``user`` message holding LangChain core's
            format instructions for ``output_schema``; ``response_format`` None.

        Example:
            >>> FormatInstructionsMethod().prepare(messages, Verdict).messages[-1][
            ...     "role"]
            'user'
        """
        parser: PydanticOutputParser[BaseModel] = PydanticOutputParser(
            pydantic_object=output_schema
        )
        instruction = {"role": "user", "content": parser.get_format_instructions()}
        return PreparedRequest([*messages, instruction], None)

    def json_text(self, answer: str) -> str:
        r"""Return the JSON object of an answer, inside a code fence or bare.

        Args:
            answer: The answer text.

        Returns:
            The object, written again as JSON; the answer stripped when it is
            not JSON, such as an object with text around it, so that the schema
            refuses it and it is asked again.

        Example:
            >>> FormatInstructionsMethod().json_text('```json\n{"ok": true}\n```')
            '{"ok": true}'
        """
        try:
            # json.loads, not LangChain's default parser: that one completes a
            # truncated object, which would then pass for a whole answer.
            found = parse_json_markdown(answer, parser=json.loads)
        except ValueError:
            return answer.strip()
        return json.dumps(found, ensure_ascii=False)
