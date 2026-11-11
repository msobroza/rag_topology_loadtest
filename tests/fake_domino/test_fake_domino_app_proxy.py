"""The fake's App proxy: who gets through, and what reaches the App behind it."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
import pytest
from fake_domino_support import (
    API_KEY_AUTH,
    INSTANCE_PATH,
    RUN_TOKEN_AUTH,
    FakeUpstream,
    create,
    http_to,
)
from fastapi.testclient import TestClient

from fake_domino.identity import FAKE_RUN_TOKEN
from fake_domino.proxy_settings import FakeDominoProxySettings, UnauthenticatedResponse
from fake_domino.state import FakeDominoApps

UPSTREAM = "http://127.0.0.1:9001"
APP_URLS = ["/apps/app-1", f"{INSTANCE_PATH}/app-1-v1-i"]


def behind_the_proxy(
    upstream: FakeUpstream,
    version: Mapping[str, object] | None = None,
    **settings: object,
) -> TestClient:
    proxy = FakeDominoProxySettings(upstreams={"rag-reranker": UPSTREAM}, **settings)  # type: ignore[arg-type]
    http = http_to(FakeDominoApps(), proxy, upstream.transport())
    create(http, "rag-reranker", version=version)
    return http


@pytest.mark.parametrize("app_url", APP_URLS)
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer not-the-token"}, {"X-Domino-Api-Key": "nope"}],
)
def test_a_request_without_a_valid_credential_is_401(
    app_url: str, headers: dict[str, str]
) -> None:
    upstream = FakeUpstream()
    response = behind_the_proxy(upstream).get(f"{app_url}/readyz", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer realm="DominoRealm"'
    assert f"{app_url}/readyz" in response.json()["detail"]
    assert upstream.seen == []


def test_the_gate_comes_before_the_app_lookup() -> None:
    response = http_to(FakeDominoApps()).get("/apps/app-9/readyz")
    assert response.status_code == 401


def test_redirect_mode_sends_an_unauthenticated_caller_to_the_login_page() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream, unauthenticated=UnauthenticatedResponse.REDIRECT)
    response = http.get("/apps/app-1/docs")
    assert response.status_code == 302
    assert response.headers["Location"] == (
        "/auth/realms/DominoRealm/protocol/openid-connect/auth?redirect_uri="
        "http%3A%2F%2Ffake-domino.test%2Fapps%2Fapp-1%2Fdocs"
    )
    assert upstream.seen == []


@pytest.mark.parametrize("app_url", APP_URLS)
@pytest.mark.parametrize("credential", [RUN_TOKEN_AUTH, API_KEY_AUTH])
def test_a_credential_reaches_the_app_and_its_answer_comes_back(
    app_url: str, credential: dict[str, str]
) -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream)
    response = http.get(f"{app_url}/v3/models", headers=credential)
    assert (response.status_code, response.text) == (201, "from the app")
    assert response.headers["X-Upstream"] == "fake-app"
    assert [str(request.url) for request in upstream.seen] == [f"{UPSTREAM}/v3/models"]


def test_method_sub_path_query_and_body_are_forwarded_without_the_prefix() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream)
    http.put(
        "/apps/app-1/v3/rerank?top_n=3&x=a%20b",
        headers=RUN_TOKEN_AUTH,
        content=b'{"query": "q"}',
    )
    [sent] = upstream.seen
    assert sent.method == "PUT"
    assert str(sent.url) == f"{UPSTREAM}/v3/rerank?top_n=3&x=a%20b"
    assert sent.content == b'{"query": "q"}'


def test_keep_prefix_forwards_the_domino_path_as_it_came() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream, keep_prefix=True)
    http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    http.get(f"{INSTANCE_PATH}/app-1-v1-i/v3/models", headers=RUN_TOKEN_AUTH)
    assert [str(request.url) for request in upstream.seen] == [
        f"{UPSTREAM}/apps/app-1/v3/models",
        f"{UPSTREAM}{INSTANCE_PATH}/app-1-v1-i/v3/models",
    ]


def test_the_callers_credentials_are_replaced_by_dominos_headers() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream)
    headers = {**API_KEY_AUTH, "domino-username": "mallory", "X-Request-ID": "r-1"}
    http.get("/apps/app-1/v3/models", headers=headers)
    [sent] = upstream.seen
    assert sent.headers.get_list("domino-username") == ["fake-owner"]
    assert "X-Domino-Api-Key" not in sent.headers
    assert sent.headers["X-Request-ID"] == "r-1"
    scheme, _, token = sent.headers["Authorization"].partition(" ")
    assert (scheme, token.count(".")) == ("Bearer", 2)  # a JWT, not the run token


def test_without_the_identity_jwt_the_app_gets_only_the_username() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream, identity_jwt=False)
    http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    [sent] = upstream.seen
    assert sent.headers["domino-username"] == "fake-owner"
    assert "Authorization" not in sent.headers


@pytest.mark.parametrize("credential", [RUN_TOKEN_AUTH, API_KEY_AUTH])
def test_extended_identity_sends_the_viewers_token_and_no_username(
    credential: dict[str, str],
) -> None:
    upstream = FakeUpstream()
    version = {"extendedIdentityPropagationToAppsEnabled": True}
    http = behind_the_proxy(upstream, version=version)
    http.get("/apps/app-1/v3/models", headers=credential)
    [sent] = upstream.seen
    assert sent.headers["Authorization"] == f"Bearer {FAKE_RUN_TOKEN}"
    assert "domino-username" not in sent.headers


def test_an_upstream_that_does_not_answer_is_502_naming_it() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    proxy = FakeDominoProxySettings(upstreams={"rag-reranker": UPSTREAM})
    http = http_to(FakeDominoApps(), proxy, httpx.MockTransport(refuse))
    create(http, "rag-reranker")
    response = http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    assert response.status_code == 502
    assert response.text == (
        f"upstream {UPSTREAM} of App 'rag-reranker' did not answer: "
        "ConnectError: connection refused"
    )


def test_the_forward_carries_the_configured_timeout() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream, forward_timeout_seconds=7.5)
    http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    [sent] = upstream.seen
    assert set(sent.extensions["timeout"].values()) == {7.5}


def test_an_app_with_no_upstream_answers_502_except_for_readyz() -> None:
    upstream = FakeUpstream()
    http = behind_the_proxy(upstream)
    create(http, "rag-workflow")  # app-2, no upstream configured
    response = http.post("/apps/app-2/v3/answer", headers=RUN_TOKEN_AUTH)
    assert response.status_code == 502
    assert response.text == (
        "no upstream configured for App 'rag-workflow' "
        "(rag-fake-domino --upstream rag-workflow=URL)"
    )
    assert http.get("/apps/app-2/readyz", headers=RUN_TOKEN_AUTH).status_code == 200
    assert upstream.seen == []


@pytest.mark.parametrize(
    ("app_url", "detail"),
    [
        ("/apps/app-9", "no App app-9"),
        (f"{INSTANCE_PATH}/app-9-v1-i", "no App instance app-9-v1-i"),
    ],
)
def test_an_unknown_app_or_instance_is_404(app_url: str, detail: str) -> None:
    response = http_to(FakeDominoApps()).get(
        f"{app_url}/v3/models", headers=RUN_TOKEN_AUTH
    )
    assert (response.status_code, response.json()) == (404, {"detail": detail})
