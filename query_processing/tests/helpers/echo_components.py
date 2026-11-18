"""A fixture component type, ``echo``, and one implementation, for plain-call tests.

An echo takes a text and gives it back with its length; ``Upper`` gives it in
capitals. The type registers in ``ECHO_TYPES``, a registry of the tests' own,
so the toolkit's ``COMPONENT_TYPES`` stays the four of the toolkit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from query_processing.domain.components.base_component import BaseComponent
from query_processing.domain.registry import Registry


class EchoRequest(BaseModel):
    text: str


class Echo(BaseModel):
    text: str
    length: int


ECHO_TYPES: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")


class BaseEcho(
    BaseComponent[EchoRequest, Echo], component_type="echo", registry=ECHO_TYPES
):
    """Answers a text with what it found in it."""


class Upper(BaseEcho):
    """Gives the text in capitals, and its length."""

    def __init__(self) -> None:
        self.asked: list[EchoRequest] = []

    def invoke(self, request: EchoRequest) -> Echo:
        self.asked.append(request)
        return Echo(text=request.text.upper(), length=len(request.text))
