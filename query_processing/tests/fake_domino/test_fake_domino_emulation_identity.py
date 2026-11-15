"""The fake's one user: its credentials, its identity JWT and the key that signs it."""

from __future__ import annotations

import jwt
import pytest
from starlette.datastructures import Headers
from test_fake_domino_emulation_support import (
    RUN_TOKEN_AUTH,
    FakeUpstream,
    create,
    http_to,
)

from fake_domino.emulation.identity import (
    FAKE_API_KEY,
    FAKE_RUN_TOKEN,
    FAKE_USERNAME,
    FakeIdentityTokens,
    caller_of,
)
from fake_domino.emulation.proxy_settings import FakeDominoProxySettings
from fake_domino.state import FakeDominoApps

JWKS_PATH = "/auth/realms/DominoRealm/protocol/openid-connect/certs"


@pytest.mark.parametrize(
    ("headers", "caller"),
    [
        ({"Authorization": f"Bearer {FAKE_RUN_TOKEN}"}, FAKE_USERNAME),
        ({"authorization": f"bearer {FAKE_RUN_TOKEN}"}, FAKE_USERNAME),
        ({"X-Domino-Api-Key": FAKE_API_KEY}, FAKE_USERNAME),
        ({"Authorization": "Bearer someone-else"}, None),
        ({"Authorization": f"Basic {FAKE_RUN_TOKEN}"}, None),
        ({"X-Domino-Api-Key": FAKE_RUN_TOKEN}, None),
        ({}, None),
    ],
)
def test_a_credential_resolves_to_the_fake_user_or_to_no_caller(
    headers: dict[str, str], caller: str | None
) -> None:
    assert caller_of(Headers(headers)) == caller


def public_key(jwks: dict[str, list[dict[str, object]]]) -> jwt.PyJWK:
    [key] = jwks["keys"]
    return jwt.PyJWK(key)


def test_a_signed_token_carries_the_documented_claims() -> None:
    tokens = FakeIdentityTokens(clock=lambda: 1_000.0)
    token = tokens.sign("fake-owner", issuer="http://d.test/auth/realms/DominoRealm")
    claims = jwt.decode(
        token,
        public_key(tokens.jwks()),
        algorithms=["RS256"],
        audience="apps",
        options={"verify_exp": False, "verify_iat": False},
    )
    assert claims == {
        "sub": "fake-owner",
        "preferred_username": "fake-owner",
        "email": "fake-owner@fake-domino.test",
        "given_name": "Fake",
        "family_name": "Owner",
        "aud": "apps",
        "iss": "http://d.test/auth/realms/DominoRealm",
        "iat": 1_000,
        "exp": 1_300,
    }
    assert jwt.get_unverified_header(token)["kid"] == "fake-domino"


def test_the_jwks_holds_only_the_public_key() -> None:
    [key] = FakeIdentityTokens().jwks()["keys"]
    assert (key["kty"], key["alg"], key["use"], key["kid"]) == (
        "RSA",
        "RS256",
        "sig",
        "fake-domino",
    )
    assert "d" not in key


def test_the_forwarded_jwt_verifies_against_the_served_jwks() -> None:
    upstream = FakeUpstream()
    proxy = FakeDominoProxySettings(upstreams={"rag-reranker": "http://app.test"})
    http = http_to(FakeDominoApps(), proxy, upstream.transport())
    create(http, "rag-reranker")
    http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    token = upstream.seen[0].headers["Authorization"].removeprefix("Bearer ")
    served = http.get(JWKS_PATH)
    assert served.status_code == 200
    claims = jwt.decode(
        token, public_key(served.json()), algorithms=["RS256"], audience="apps"
    )
    assert claims["preferred_username"] == upstream.seen[0].headers["domino-username"]
    assert claims["iss"] == "http://fake-domino.test/auth/realms/DominoRealm"


def test_the_jwks_route_needs_no_credential() -> None:
    http = http_to(FakeDominoApps.from_behaviour("forbidden"))
    assert http.get(JWKS_PATH).status_code == 200
