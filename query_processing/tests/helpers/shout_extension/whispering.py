"""A second component type of the tests' own, ``whisperer``, that redefines what the toolkit builds.

It registers an API format of its own, whose wire names the text ``words``,
and a ``served`` class of its own, which marks what the model server answered:
so a test can tell them from the ones the toolkit builds for ``shouter``.
"""

from __future__ import annotations

from typing import Any

from query_processing.extension_api import (
    COMPONENT_API_FORMATS,
    BaseApiFormat,
    BaseComponent,
    ServedCallMixin,
)

from .shouting import Shouted, ShoutRequest

# What the served class adds to each answer, so a test sees it was used.
SERVED_MARK = " (served)"


class BaseWhisperer(BaseComponent[ShoutRequest, Shouted], component_type="whisperer"):
    """Says a text softer."""


class LowerWhisperer(BaseWhisperer, backend="local"):
    """Says the text in small letters."""

    def invoke(self, request: ShoutRequest) -> Shouted:
        return Shouted(text=request.text.lower())


class MarkingServedWhisperer(ServedCallMixin, BaseWhisperer, backend="served"):
    """Calls the model server as the built class would, and marks the answer."""

    async def ainvoke(self, request: ShoutRequest) -> Shouted:
        answer: Shouted = await super().ainvoke(request)
        return Shouted(text=answer.text + SERVED_MARK)


@COMPONENT_API_FORMATS.register("whisperer")
class WordsFormat(BaseApiFormat[ShoutRequest, Shouted]):
    """The text under ``words``, both ways."""

    def request_to_wire(self, request: ShoutRequest) -> dict[str, object]:
        return {"words": request.text}

    def request_from_wire(self, body: object) -> ShoutRequest:
        return ShoutRequest(text=_words(body))

    def answer_to_wire(self, answer: Shouted) -> dict[str, object]:
        return {"words": answer.text}

    def answer_from_wire(self, body: object) -> Shouted:
        return Shouted(text=_words(body))


def _words(body: Any) -> str:
    if not isinstance(body, dict) or not isinstance(body.get("words"), str):
        raise ValueError(f"body {body!r}; expected {{'words': <text>}}")
    return body["words"]


class BaseMurmurer(BaseComponent[ShoutRequest, Shouted], component_type="murmurer"):
    """Says a text under the breath: a type with no ``local`` backend."""


class FakeMurmurer(BaseMurmurer, backend="fake"):
    """Says the text as it is."""

    def invoke(self, request: ShoutRequest) -> Shouted:
        return Shouted(text=request.text)
