"""``BaseApiFormat``: a component type's request and answer to the wire and back.

A request and an answer are domain classes and never go on the wire as they
are: an API format translates each in both directions, so the class that calls
a model server and the model server's call route speak the same body. A
component type that registers no format in ``COMPONENT_API_FORMATS`` gets the
one the toolkit builds, which mirrors the request and the answer: their fields,
under their names, as JSON values. It reads a body with the domain class
itself, so it refuses on the wire whatever the class refuses.

Example:
    >>> api_format = api_format_of(BaseShouter)
    >>> api_format.request_to_wire(ShoutRequest(text="a"))
    {'text': 'a'}
    >>> api_format.answer_from_wire({"text": "A"})
    Shouted(text='A')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter

from ..components.base_component import BaseComponent
from ..registry import Registry

RequestT = TypeVar("RequestT")
AnswerT = TypeVar("AnswerT")


class BaseApiFormat(ABC, Generic[RequestT, AnswerT]):
    """How one component type's request and answer look on the wire.

    Each ``*_from_wire`` raises ``ValueError`` for a body that is not its
    shape, naming what it found and what it expected: the model server answers
    a request it cannot read with a 422, and the served class an answer it
    cannot read with the dependency error that names the component.

    Example:
        >>> @COMPONENT_API_FORMATS.register("whisperer")
        ... class WordsFormat(BaseApiFormat[ShoutRequest, Shouted]): ...
    """

    @abstractmethod
    def request_to_wire(self, request: RequestT) -> dict[str, object]:
        """Return the JSON body that carries the request.

        Args:
            request: The component type's request.

        Returns:
            The body, JSON values only.

        Example:
            >>> api_format.request_to_wire(ShoutRequest(text="a"))
            {'text': 'a'}
        """

    @abstractmethod
    def request_from_wire(self, body: object) -> RequestT:
        """Return the request a JSON body carries.

        Args:
            body: The decoded JSON body.

        Returns:
            The component type's request.

        Raises:
            ValueError: If the body is not the request's shape.

        Example:
            >>> api_format.request_from_wire({"text": "a"})
            ShoutRequest(text='a')
        """

    @abstractmethod
    def answer_to_wire(self, answer: AnswerT) -> dict[str, object]:
        """Return the JSON body that carries the answer.

        Args:
            answer: The component type's answer.

        Returns:
            The body, JSON values only.

        Example:
            >>> api_format.answer_to_wire(Shouted(text="A"))
            {'text': 'A'}
        """

    @abstractmethod
    def answer_from_wire(self, body: object) -> AnswerT:
        """Return the answer a JSON body carries.

        Args:
            body: The decoded JSON body.

        Returns:
            The component type's answer.

        Raises:
            ValueError: If the body is not the answer's shape.

        Example:
            >>> api_format.answer_from_wire({"text": "A"})
            Shouted(text='A')
        """


class MirroredApiFormat(BaseApiFormat[Any, Any]):
    """The format the toolkit builds: the request's and the answer's own fields.

    Args:
        request_type: The component type's request class.
        answer_type: Its answer class.

    Example:
        >>> MirroredApiFormat(ShoutRequest, Shouted).request_to_wire(
        ...     ShoutRequest(text="a"))
        {'text': 'a'}
    """

    def __init__(self, request_type: type[Any], answer_type: type[Any]) -> None:
        # The domain class's own validation, so nothing it refuses gets through.
        self._request = TypeAdapter(request_type)
        self._answer = TypeAdapter(answer_type)

    def request_to_wire(self, request: Any) -> dict[str, object]:
        """Return the request's fields, under their names, as JSON values.

        Args:
            request: The component type's request.

        Returns:
            The body.

        Example:
            >>> api_format.request_to_wire(ShoutRequest(text="a"))
            {'text': 'a'}
        """
        return _as_body(self._request, request)

    def request_from_wire(self, body: object) -> Any:
        """Read the request with its own class, its validators included.

        Args:
            body: The decoded JSON body.

        Returns:
            The component type's request.

        Raises:
            ValueError: If the request's class refuses the body (a ValidationError).

        Example:
            >>> api_format.request_from_wire({"text": "a"})
            ShoutRequest(text='a')
        """
        return self._request.validate_python(body)

    def answer_to_wire(self, answer: Any) -> dict[str, object]:
        """Return the answer's fields, under their names, as JSON values.

        Args:
            answer: The component type's answer.

        Returns:
            The body.

        Example:
            >>> api_format.answer_to_wire(Shouted(text="A"))
            {'text': 'A'}
        """
        return _as_body(self._answer, answer)

    def answer_from_wire(self, body: object) -> Any:
        """Read the answer with its own class, its validators included.

        Args:
            body: The decoded JSON body.

        Returns:
            The component type's answer.

        Raises:
            ValueError: If the answer's class refuses the body (a ValidationError).

        Example:
            >>> api_format.answer_from_wire({"text": "A"})
            Shouted(text='A')
        """
        return self._answer.validate_python(body)


def _as_body(adapter: TypeAdapter[Any], value: object) -> dict[str, object]:
    body = adapter.dump_python(value, mode="json")
    if not isinstance(body, dict):
        raise TypeError(
            f"{value!r} goes on the wire as {body!r}; expected an object of "
            "fields: a request and an answer are classes with fields"
        )
    return body


def api_format_of(
    component_type: type[BaseComponent[Any, Any]],
) -> BaseApiFormat[Any, Any]:
    """Return the API format of a component type: its registered one, else the mirror.

    Args:
        component_type: The component type's abstract class.

    Returns:
        A new instance of the format registered under its type name in
        ``COMPONENT_API_FORMATS``, else a ``MirroredApiFormat`` of its request
        and its answer.

    Example:
        >>> api_format_of(BaseShouter).request_to_wire(ShoutRequest(text="a"))
        {'text': 'a'}
    """
    name = component_type.component_type_name
    if name in COMPONENT_API_FORMATS.names():
        return COMPONENT_API_FORMATS.get(name)()
    return MirroredApiFormat(component_type.request_type, component_type.answer_type)


COMPONENT_API_FORMATS: Registry[type[BaseApiFormat[Any, Any]]] = Registry(
    "component API format"
)
"""The API format each component type registers, by its name, in place of the mirror."""
