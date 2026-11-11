"""The fake App proxy's options: one frozen value handed to ``create_fake_domino``.

Each option is a choice Domino's documentation leaves open, so the fake can be
run either way; ``rag-fake-domino`` fills them from its flags.

Example:
    >>> create_fake_domino(proxy=FakeDominoProxySettings(keep_prefix=True))
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class UnauthenticatedResponse(StrEnum):
    """What the proxy answers a request with no valid credential (undocumented)."""

    UNAUTHORIZED = "401"  # a script-facing proxy: 401 with WWW-Authenticate
    REDIRECT = "redirect"  # a browser-facing proxy: 302 to the login page


@dataclass(frozen=True)
class FakeDominoProxySettings:
    """How the fake wraps an App the way Domino's App proxy does.

    Attributes:
        unauthenticated: The answer to a request with no valid credential.
        identity_jwt: Whether secure identity propagation is on: the App gets
            ``Authorization: Bearer <identity JWT>`` besides ``domino-username``.
        keep_prefix: Forward ``/apps/<id>/x`` as is; False forwards ``/x``.
        upstreams: Base URL of the running App, by App name, e.g.
            ``{"rag-reranker": "http://127.0.0.1:9001"}``.
        forward_timeout_seconds: Timeout of each forwarded request.

    Example:
        >>> FakeDominoProxySettings(upstreams={"rag-reranker": "http://127.0.0.1:9001"})
    """

    unauthenticated: UnauthenticatedResponse = UnauthenticatedResponse.UNAUTHORIZED
    identity_jwt: bool = True
    keep_prefix: bool = False
    upstreams: Mapping[str, str] = field(default_factory=dict)
    forward_timeout_seconds: float = 30.0
