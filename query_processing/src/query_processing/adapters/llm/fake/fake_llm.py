"""The ``fake`` LLM: echoes the last user message, after an optional latency.

Under an output schema it answers a valid instance of the schema, its first
text field holding the echo, so a workflow that asks for a schema runs with no
model.

Its token counts are word counts: no tokenizer is loaded, and a load test of
the fake LLM only needs counts that grow with the text.

Example:
    >>> await FakeLLM().generate([{"role": "user", "content": "Hi"}])
    GenerateAnswer(text='[fake-llm] Hi', prompt_tokens=1, completion_tokens=2)
"""

from __future__ import annotations

import asyncio
import types
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ....domain.component_requests import GenerateAnswer
from ..base_llm import BaseLLM

# The value of each plain type of a field the fake fills; a field of another
# type gets a value by _fake_value's rules.
_PLAIN_VALUES: dict[object, object] = {str: "", bool: False, int: 0, float: 0.0}
_EMPTY_CONTAINERS: dict[object, object] = {list: [], set: [], frozenset: [], dict: {}}


class FakeLLM(BaseLLM):
    """Echoes the last user message; ``latency_ms`` imitates generation time.

    Args:
        latency_ms: Milliseconds each ``generate`` sleeps first; 0 answers at once.
    """

    model_name = "fake-llm"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms, self.calls = latency_ms, 0

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        """Echo the last user message, after ``latency_ms``; counts the calls.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            output_schema: The pydantic model the answer must follow; None for
                the echo as text.
            max_tokens: Ignored: the echo is never cut, so an answer under a
                schema stays an instance of it.

        Returns:
            ``"[fake-llm] "`` and the first 200 characters of the last user message
            ("" when there is none). Under ``output_schema``, the JSON of an
            instance it accepts: its first required text field holds the echo,
            each list and dict is empty, each optional is null, and a field
            with a default keeps it. As token counts, the whitespace-separated
            words of every message's content and of the answer's text.

        Raises:
            TypeError: A required field of the schema has a type the fake
                cannot fill; the message names the field and its type.

        Example:
            >>> (await FakeLLM().generate(messages, output_schema=Verdict)).text
            '{"ok":false}'
        """
        del max_tokens  # the port's signature; the echo is never cut
        self.calls += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        echo = f"[fake-llm] {user[:200]}"
        text = echo
        if output_schema is not None:
            text = fake_instance_json(output_schema, echo)
        prompt_words = sum(len(message["content"].split()) for message in messages)
        return GenerateAnswer(
            text=text, prompt_tokens=prompt_words, completion_tokens=len(text.split())
        )


def fake_instance_json(schema: type[BaseModel], text: str) -> str:
    """Return the JSON of an instance of ``schema`` that holds ``text``.

    Args:
        schema: The pydantic model to fill.
        text: What its first required ``str`` field holds.

    Returns:
        The instance's JSON, by alias, which the schema accepts.

    Raises:
        TypeError: A required field has a type the fake cannot fill.

    Example:
        >>> fake_instance_json(Rewritten, "q")
        '{"standalone_query":"q","metadata_filter":[]}'
    """
    values = _required_values(schema)
    first_text = next(
        (key for key, info in _required(schema) if info.annotation is str), None
    )
    if first_text is not None:
        values[first_text] = text
    return schema.model_validate(values).model_dump_json(by_alias=True)


def _required(schema: type[BaseModel]) -> list[tuple[str, FieldInfo]]:
    # Each required field, keyed as validation reads it: by its alias if it has one.
    fields = schema.model_fields.items()
    return [(info.alias or name, info) for name, info in fields if info.is_required()]


def _required_values(schema: type[BaseModel]) -> dict[str, object]:
    # A field with a default is left out, so the schema fills it.
    return {key: _fake_value(key, info.annotation) for key, info in _required(schema)}


def _fake_value(name: str, annotation: Any) -> object:
    # The simplest value of a field's type that the type accepts.
    if annotation in _PLAIN_VALUES:
        return _PLAIN_VALUES[annotation]
    origin = get_origin(annotation)
    if origin in _EMPTY_CONTAINERS:
        return _EMPTY_CONTAINERS[origin]
    if origin is Literal:
        return get_args(annotation)[0]
    if origin in (Union, types.UnionType):
        return _union_value(name, get_args(annotation))
    if isinstance(annotation, type):
        return _class_value(name, annotation)
    raise _unfillable(name, annotation)


def _union_value(name: str, members: tuple[Any, ...]) -> object:
    # An optional is null; another union takes the value of its first member.
    if type(None) in members:
        return None
    return _fake_value(name, members[0])


def _class_value(name: str, annotation: type) -> object:
    if issubclass(annotation, Enum):
        return next(iter(annotation)).value
    if issubclass(annotation, BaseModel):
        return _required_values(annotation)
    if annotation in _EMPTY_CONTAINERS:
        return _EMPTY_CONTAINERS[annotation]
    raise _unfillable(name, annotation)


def _unfillable(name: str, annotation: object) -> TypeError:
    return TypeError(
        f"the fake LLM cannot fill the field {name!r} of type {annotation!r} "
        "(expected str, bool, int, float, a list, a set, a dict, a Literal, an "
        "Enum, an optional, a union or a pydantic model)"
    )
