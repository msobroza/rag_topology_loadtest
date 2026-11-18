"""The router's API format, written by hand: the worked example of one.

No outside API format exists for a router, so it speaks one of our own on its
call route: the query's vector under ``embedding``, and the route choice as
``route`` and ``confidence``. It registers under ``router`` in
``COMPONENT_API_FORMATS``, in place of the mirror the toolkit would build
(which would name the route ``name``); a use case registers its own format
for a component type of its own the same way.

Example:
    >>> RouterApiFormat().answer_to_wire(RouteChoice(name="a", confidence=0.5))
    {'route': 'a', 'confidence': 0.5}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from ...domain.components.routes import RouteChoice, RouteRequest
from ...domain.endpoints.api_format import COMPONENT_API_FORMATS, BaseApiFormat

_REQUEST_SHAPE = "{'embedding': [<float>, ...]}"
_ANSWER_SHAPE = "{'route': <name>, 'confidence': <float>}"


@COMPONENT_API_FORMATS.register("router")
class RouterApiFormat(BaseApiFormat[RouteRequest, RouteChoice]):
    """A query's vector in, the route chosen and its confidence out."""

    def request_to_wire(self, request: RouteRequest) -> dict[str, object]:
        """Return the body that carries the query's vector.

        Args:
            request: The router's request.

        Returns:
            ``{"embedding": [...]}``.

        Example:
            >>> RouterApiFormat().request_to_wire(RouteRequest(embedding=[0.1]))
            {'embedding': [0.1]}
        """
        return {"embedding": list(request.embedding)}

    def request_from_wire(self, body: object) -> RouteRequest:
        """Return the request a body carries.

        Args:
            body: The decoded JSON body.

        Returns:
            The router's request.

        Raises:
            ValueError: If the body is not ``{"embedding": [...]}``.

        Example:
            >>> RouterApiFormat().request_from_wire({"embedding": [0.1]})
            RouteRequest(embedding=[0.1])
        """
        keys = _keys(body, {"embedding"}, _REQUEST_SHAPE)
        return _read(RouteRequest, keys, body, _REQUEST_SHAPE)

    def answer_to_wire(self, answer: RouteChoice) -> dict[str, object]:
        """Return the body that carries the route choice, the route under ``route``.

        Args:
            answer: The router's answer.

        Returns:
            ``{"route": ..., "confidence": ...}``.

        Example:
            >>> RouterApiFormat().answer_to_wire(RouteChoice(name="a", confidence=1.0))
            {'route': 'a', 'confidence': 1.0}
        """
        return {"route": answer.name, "confidence": answer.confidence}

    def answer_from_wire(self, body: object) -> RouteChoice:
        """Return the route choice a body carries.

        Args:
            body: The decoded JSON body.

        Returns:
            The router's answer.

        Raises:
            ValueError: If the body is not ``{"route": ..., "confidence": ...}``.

        Example:
            >>> RouterApiFormat().answer_from_wire({"route": "a", "confidence": 1.0})
            RouteChoice(name='a', confidence=1.0)
        """
        keys = _keys(body, {"route", "confidence"}, _ANSWER_SHAPE)
        fields = {"name": keys["route"], "confidence": keys["confidence"]}
        return _read(RouteChoice, fields, body, _ANSWER_SHAPE)


def _keys(body: object, expected: set[str], shape: str) -> Mapping[str, Any]:
    if not isinstance(body, Mapping) or set(body) != expected:
        raise ValueError(f"body {body!r}; expected {shape}")
    return body


def _read(model: type[Any], fields: Mapping[str, Any], body: object, shape: str) -> Any:
    try:
        return model.model_validate(fields)
    except ValidationError as exc:
        raise ValueError(f"body {body!r}; expected {shape}: {exc}") from exc
