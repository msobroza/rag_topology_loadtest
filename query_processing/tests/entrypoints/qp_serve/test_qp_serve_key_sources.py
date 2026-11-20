"""An API key reaches each call through a key source, seen from outside the workflow app.

A served component on Domino is called with the run token of the process's one
``domino`` source; an ``api`` component with its fixed key or with none. When a
call is refused for its key (401), its source is told and the call is made
again within the retries of the ``http`` block; a 403 and a 401 with no key
fail at once. The endpoints and Domino's token URL are one HTTP transport that
answers in their place, numbers the run tokens it serves, refuses the keys it is
told to refuse, and records what each call carried.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from model_api_fakes import standard_answer
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app

SettingsFactory = Callable[..., WorkflowSettings]
QUESTION = "How many paid vacation days do employees accrue each year?"
EMBED_URL = "http://embedder-app.test/v3/embeddings"
RERANK_URL = "http://reranker-app.test/v3/rerank"
VENDOR_RERANK_URL = "https://rerank.vendor.test/v2/rerank"
TOKEN_URL = "http://domino.test/access-token"
FIXED_KEY = "co-secret-key-4711"
# Every component on its fake but the reranker, called at RERANK_URL; no wait
# between retries, two retries.
SERVED_RERANKER: dict[str, object] = {
    "components__embedder__backend": "fake",
    "components__reranker__backend": "served",
    "components__reranker__served__url": RERANK_URL,
    "components__llm__backend": "fake",
    "components__vector_store__backend": "fake",
    "domino__token_url": TOKEN_URL,
    "http__retries": 2,
    "http__backoff_s": 0.0,
}
VENDOR_RERANKER: dict[str, object] = {
    "components__reranker__backend": "api",
    "components__reranker__api__url": VENDOR_RERANK_URL,
    "components__reranker__api__api_format": "cohere",
    "components__reranker__api__model": "rerank-v3.5",
}


def run_token(number: int) -> str:
    """The ``number``-th run token the stand-in serves, from 1."""
    return f"run-token-{number}"


class KeyedEndpoints:
    """Every endpoint and Domino's token URL, one refusing the keys it is told to.

    The token URL answers its scripted failures first, then numbered run tokens.
    An endpoint answers ``refusal`` to a call whose ``Authorization`` is in
    ``refused`` (``None`` for a call with no key), else its standard answer.
    """

    def __init__(
        self,
        *,
        refused: tuple[str | None, ...] = (),
        refusal: int = 401,
        token_failures: tuple[httpx.Response | Exception, ...] = (),
    ) -> None:
        self.refused, self.refusal = set(refused), refusal
        self.token_failures = list(token_failures)
        self.tokens_served = 0
        self.token_asks = 0
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return self._token()
        auth = request.headers.get("Authorization")
        self.calls.append((str(request.url), auth))
        if auth in self.refused:
            return httpx.Response(self.refusal, text="key refused")
        return httpx.Response(200, json=standard_answer(json.loads(request.content)))

    def _token(self) -> httpx.Response:
        self.token_asks += 1
        if self.token_failures:
            failure = self.token_failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return failure
        self.tokens_served += 1
        return httpx.Response(200, text=run_token(self.tokens_served))

    def sent_to(self, url: str) -> list[str | None]:
        """The ``Authorization`` of every call made to ``url``, in order."""
        return [auth for called, auth in self.calls if called == url]


def app_over(
    endpoints: KeyedEndpoints,
    settings_factory: SettingsFactory,
    *,
    on_domino: bool,
    **overrides: object,
) -> TestClient:
    """The workflow app on ``SERVED_RERANKER`` and ``overrides``, its calls to ``endpoints``."""
    settings = settings_factory(**(SERVED_RERANKER | overrides))
    transport = httpx.MockTransport(endpoints)
    composition = CompositionRoot(settings, transport=transport, on_domino=on_domino)
    return TestClient(create_app(composition))


def retrieve(client: TestClient) -> httpx.Response:
    return client.post("/retrieve", json={"question": QUESTION})


# --- the run token: one domino source a process, held for its time -------------------


def test_two_served_components_on_domino_share_one_run_token(
    settings_factory: SettingsFactory,
) -> None:
    # Changed on purpose: each endpoint asked for a token of its own before.
    endpoints = KeyedEndpoints()
    served_embedder = {
        "components__embedder__backend": "served",
        "components__embedder__served__url": EMBED_URL,
    }
    app = app_over(endpoints, settings_factory, on_domino=True, **served_embedder)

    with app as client:
        assert retrieve(client).status_code == 200

    assert endpoints.tokens_served == 1
    sent = endpoints.sent_to(EMBED_URL) + endpoints.sent_to(RERANK_URL)
    assert set(sent) == {f"Bearer {run_token(1)}"}
    assert endpoints.sent_to(EMBED_URL)
    assert endpoints.sent_to(RERANK_URL)


@pytest.mark.parametrize(
    ("ttl_s", "tokens"), [(240.0, 1), (1e-9, 2)], ids=["held", "asked-again"]
)
def test_the_run_token_is_held_for_its_ttl_then_asked_for_again(
    settings_factory: SettingsFactory, ttl_s: float, tokens: int
) -> None:
    endpoints = KeyedEndpoints()
    app = app_over(
        endpoints, settings_factory, on_domino=True, domino__token_ttl_s=ttl_s
    )

    with app as client:
        assert retrieve(client).status_code == 200
        assert retrieve(client).status_code == 200

    assert endpoints.tokens_served == tokens
    expected = [f"Bearer {run_token(n)}" for n in (1, tokens)]
    assert endpoints.sent_to(RERANK_URL) == expected


def test_a_refused_run_token_is_asked_for_again_and_the_client_gets_its_answer(
    settings_factory: SettingsFactory,
) -> None:
    endpoints = KeyedEndpoints(refused=(f"Bearer {run_token(1)}",))
    app = app_over(endpoints, settings_factory, on_domino=True)

    with app as client:
        response = retrieve(client)

    assert response.status_code == 200, response.text
    assert endpoints.sent_to(RERANK_URL) == [
        f"Bearer {run_token(1)}",
        f"Bearer {run_token(2)}",
    ]
    assert endpoints.tokens_served == 2


# --- asking Domino for the token -------------------------------------------------------


def test_asking_for_the_run_token_is_retried_within_the_http_block(
    settings_factory: SettingsFactory,
) -> None:
    # Changed on purpose: the token was asked for once before.
    failures = (httpx.Response(500, text="nope"), httpx.ConnectError("refused"))
    endpoints = KeyedEndpoints(token_failures=failures)
    app = app_over(endpoints, settings_factory, on_domino=True)

    with app as client:
        response = retrieve(client)

    assert response.status_code == 200, response.text
    assert endpoints.token_asks == 3
    assert endpoints.sent_to(RERANK_URL) == [f"Bearer {run_token(1)}"]


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500, text="nope"),
        httpx.Response(200, text="   "),
        httpx.ConnectError("refused"),
    ],
    ids=["non-200", "empty-body", "transport-error"],
)
def test_a_run_token_that_cannot_be_had_fails_the_request_naming_its_url(
    settings_factory: SettingsFactory, failure: httpx.Response | Exception
) -> None:
    endpoints = KeyedEndpoints(token_failures=(failure,) * 3)
    app = app_over(endpoints, settings_factory, on_domino=True)

    with app as client:
        response = retrieve(client)

    assert response.status_code == 503
    body = response.json()
    assert (body["error"], body["target"]) == (
        "domino_access_token_unavailable",
        TOKEN_URL,
    )
    assert endpoints.token_asks == 3  # the first ask and the http block's 2 retries
    assert endpoints.calls == []


# --- a fixed key, no key, and a 403 ------------------------------------------------------


def test_a_refused_fixed_key_runs_out_its_retries_then_fails(
    settings_factory: SettingsFactory,
) -> None:
    # Changed on purpose: a fixed key's 401 failed at once before.
    endpoints = KeyedEndpoints(refused=(f"Bearer {FIXED_KEY}",))
    app = app_over(
        endpoints,
        settings_factory,
        on_domino=True,
        **VENDOR_RERANKER,
        components__reranker__api__api_key=FIXED_KEY,
    )

    with app as client:
        response = retrieve(client)

    assert response.status_code == 503
    body = response.json()
    assert (body["error"], body["target"]) == (
        "reranker_unavailable",
        VENDOR_RERANK_URL,
    )
    assert "HTTP 401" in body["detail"]
    assert endpoints.sent_to(VENDOR_RERANK_URL) == [f"Bearer {FIXED_KEY}"] * 3
    assert endpoints.token_asks == 0


@pytest.mark.parametrize(
    ("api_key", "refusal", "authorization"),
    [
        ("", 401, None),
        ("", 403, None),
        (FIXED_KEY, 403, f"Bearer {FIXED_KEY}"),
        ("domino", 403, f"Bearer {run_token(1)}"),
    ],
    ids=["401-no-key", "403-no-key", "403-fixed-key", "403-run-token"],
)
def test_a_403_and_a_401_without_a_key_fail_at_once(
    settings_factory: SettingsFactory,
    api_key: str,
    refusal: int,
    authorization: str | None,
) -> None:
    endpoints = KeyedEndpoints(refused=(authorization,), refusal=refusal)
    app = app_over(
        endpoints,
        settings_factory,
        on_domino=False,
        **VENDOR_RERANKER,
        components__reranker__api__api_key=api_key,
    )

    with app as client:
        response = retrieve(client)

    assert response.status_code == 503
    assert f"HTTP {refusal}" in response.json()["detail"]
    assert endpoints.sent_to(VENDOR_RERANK_URL) == [authorization]


def test_no_refusal_and_no_log_line_holds_the_key(
    settings_factory: SettingsFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    endpoints = KeyedEndpoints(refused=(f"Bearer {FIXED_KEY}",))
    app = app_over(
        endpoints,
        settings_factory,
        on_domino=False,
        **VENDOR_RERANKER,
        components__reranker__api__api_key=FIXED_KEY,
    )

    with app as client:
        refused = retrieve(client)
        probed = client.get("/readyz")

    assert refused.status_code == 503
    assert probed.status_code == 503
    captured = capsys.readouterr()
    for text in (refused.text, probed.text, captured.out, captured.err):
        assert FIXED_KEY not in text


# --- the readiness probe: one attempt, a refused key forgotten --------------------------


def test_a_readiness_probe_makes_one_attempt_and_forgets_a_refused_run_token(
    settings_factory: SettingsFactory,
) -> None:
    endpoints = KeyedEndpoints(refused=(f"Bearer {run_token(1)}",))
    app = app_over(endpoints, settings_factory, on_domino=True)

    with app as client:
        first, second = client.get("/readyz"), client.get("/readyz")

    assert (first.status_code, second.status_code) == (503, 200)
    assert endpoints.sent_to(RERANK_URL) == [
        f"Bearer {run_token(1)}",
        f"Bearer {run_token(2)}",
    ]


def test_a_readiness_probe_reports_a_run_token_that_cannot_be_had(
    settings_factory: SettingsFactory,
) -> None:
    endpoints = KeyedEndpoints(token_failures=(httpx.Response(500, text="nope"),) * 3)
    app = app_over(endpoints, settings_factory, on_domino=True)

    with app as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    [check] = [c for c in response.json()["checks"] if c["name"] == "reranker"]
    assert not check["ok"]
    assert "domino_access_token" in check["detail"]
    assert endpoints.calls == []
