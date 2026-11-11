"""The fake's one user, its credentials, and the identity JWT its App proxy signs.

Documented by Domino: the Domino API takes ``Authorization: Bearer <token>`` or
``X-Domino-Api-Key``; a run's token service (``/access-token``) answers the raw
token, carrying the publisher's identity inside an App; with secure identity
propagation the App proxy passes an RS256 JWT for the viewer, audience ``apps``,
with ``sub``, ``preferred_username``, ``email``, ``given_name`` and
``family_name``, its keys at ``/auth/realms/DominoRealm/protocol/openid-connect/certs``.

Assumed: the same two credentials open an App URL from a script; the JWT's
``iss`` is the realm URL, it lives ``IDENTITY_TOKEN_SECONDS`` and has key id
``fake-domino``; the user's e-mail and names are made up from its username.

Example:
    >>> caller_of(request.headers)  # "fake-owner", or None for no credential
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from functools import cached_property

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

FAKE_USERNAME = "fake-owner"  # the fake's one user, who owns every App
FAKE_RUN_TOKEN = "fake-run-token"
FAKE_API_KEY = "fake-api-key"
API_KEY_HEADER = "X-Domino-Api-Key"
IDENTITY_AUDIENCE = "apps"
KEYCLOAK_REALM_PATH = "/auth/realms/DominoRealm"
JWKS_PATH = f"{KEYCLOAK_REALM_PATH}/protocol/openid-connect/certs"
IDENTITY_TOKEN_SECONDS = 300
_KEY_ID = "fake-domino"
_ALGORITHM = "RS256"
_RSA_PUBLIC_EXPONENT = 65537
_RSA_KEY_BITS = 2048


def caller_of(headers: Mapping[str, str]) -> str | None:
    """Resolve a request's credential to the fake's user.

    Args:
        headers: The request's headers, looked up case-insensitively
            (``request.headers``).

    Returns:
        ``FAKE_USERNAME`` for ``Authorization: Bearer FAKE_RUN_TOKEN`` or
        ``X-Domino-Api-Key: FAKE_API_KEY``; None for anything else.

    Example:
        >>> caller_of(Headers({"X-Domino-Api-Key": FAKE_API_KEY}))
        'fake-owner'
    """
    scheme, _, token = headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token == FAKE_RUN_TOKEN:
        return FAKE_USERNAME
    if headers.get(API_KEY_HEADER) == FAKE_API_KEY:
        return FAKE_USERNAME
    return None


class FakeIdentityTokens:
    """Signs the identity JWT the App proxy hands an App, and publishes its key.

    The RSA key pair is made on first use, so building a fake stays cheap.

    Args:
        clock: Seconds since the epoch, for ``iat`` and ``exp``.

    Example:
        >>> FakeIdentityTokens().sign("fake-owner", issuer=realm_url)
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock

    @cached_property
    def _private_key(self) -> rsa.RSAPrivateKey:
        return rsa.generate_private_key(_RSA_PUBLIC_EXPONENT, _RSA_KEY_BITS)

    def sign(self, username: str, issuer: str) -> str:
        """Sign the JWT Domino's App proxy passes for ``username``.

        Args:
            username: The viewer, e.g. ``"fake-owner"``.
            issuer: The ``iss`` claim: the Keycloak realm's URL.

        Returns:
            The compact RS256 JWT, audience ``apps``.
        """
        given, _, family = username.partition("-")
        issued = int(self._clock())
        claims = {
            "sub": username,
            "preferred_username": username,
            "email": f"{username}@fake-domino.test",
            "given_name": given.title(),
            "family_name": family.title(),
            "aud": IDENTITY_AUDIENCE,
            "iss": issuer,
            "iat": issued,
            "exp": issued + IDENTITY_TOKEN_SECONDS,
        }
        headers = {"kid": _KEY_ID}
        return jwt.encode(claims, self._private_key, _ALGORITHM, headers=headers)

    def jwks(self) -> dict[str, list[dict[str, object]]]:
        """Publish the public key, as Keycloak's ``certs`` endpoint does.

        Returns:
            A JSON Web Key Set with the one RSA public key.
        """
        public = RSAAlgorithm.to_jwk(self._private_key.public_key(), as_dict=True)
        key = {**public, "kid": _KEY_ID, "alg": _ALGORITHM, "use": "sig"}
        return {"keys": [key]}
