"""How patiently the workflow app calls a component on ``served`` or ``api``, seen
through its HTTP API.

Each model component has an ``http`` block: the timeout of one attempt, the
retries after the first, and the backoff before the first retry. A value the
block leaves out is the default shared by every component (``QP_HTTP__TIMEOUT_S``,
``QP_HTTP__RETRIES``, ``QP_HTTP__BACKOFF_S``); the tests' workflow file gives
the LLM its own longer timeout. The endpoints are one HTTP transport that answers
in their place: it refuses the calls made about the question a set number of
times, and records each with its read timeout and the moment it came.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise

import httpx
import pytest
from fastapi.testclient import TestClient

from model_api_fakes import standard_answer
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS, shell_settings

SettingsFactory = Callable[..., WorkflowSettings]
# No record of the synthetic corpus holds it, so the calls that seed the fake
# store are never about the question.
QUESTION = "Which public holidays are paid at double rate this year?"
URLS = {
    "embedder": "http://embedder-app.test/v3/embeddings",
    "reranker": "http://reranker-app.test/v3/rerank",
    "llm": "http://llm-app.test/v3/chat/completions",
}
ALWAYS = 1_000  # more failures than any test lets a component attempt
TODAYS_BACKOFFS = [0.2, 0.4]  # QP_HTTP__BACKOFF_S, doubled before the 2nd retry


@dataclass(frozen=True)
class Attempt:
    """One call about the question: where it went, its read timeout, when it came."""

    url: str
    timeout_s: float
    at_s: float  # time.monotonic() as the transport received it


class FlakyModelEndpoints:
    """Every endpoint, refusing the calls about the question a set number of times.

    A call is about the question when its body holds it: the query's embedding,
    the rerank request and the chat request. At each URL the first calls about
    the question are answered 503, as many as ``failures`` says, and every later
    one in the standard format; the records that seed the fake store at
    start-up are always answered.

    Args:
        failures: By URL, how many calls about the question are refused; a URL
            not named refuses none.
    """

    def __init__(self, failures: Mapping[str, int]) -> None:
        self._failures = failures
        self.attempts: list[Attempt] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if QUESTION not in request.content.decode():
            return httpx.Response(200, json=standard_answer(body))
        url = str(request.url)
        read_timeout = request.extensions["timeout"]["read"]
        self.attempts.append(Attempt(url, read_timeout, time.monotonic()))
        if len(self.at(url)) <= self._failures.get(url, 0):
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=standard_answer(body))

    def at(self, url: str) -> list[Attempt]:
        """Every attempt about the question made at ``url``, in order."""
        return [attempt for attempt in self.attempts if attempt.url == url]


@pytest.fixture(autouse=True)
def _no_settings_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every default is the code's or the tests' file's, whatever the shell sets."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)


def called_over_http(component: str, backend: str) -> dict[str, object]:
    """Keywords putting ``component`` on ``backend`` at its URL, the others on fake."""
    return FAKE_BACKENDS | {
        f"components__{component}__backend": backend,
        f"components__{component}__{backend}__url": URLS[component],
        "components__llm__alias": "llama-3",  # a served LLM has no source id
    }


def ask(settings: WorkflowSettings, endpoints: FlakyModelEndpoints) -> httpx.Response:
    """Start the workflow app over ``endpoints`` and ask ``/query`` once."""
    composition = CompositionRoot(settings, transport=httpx.MockTransport(endpoints))
    with TestClient(create_app(composition)) as client:
        return client.post("/query", json={"question": QUESTION})


def set_environment(monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


def waited_at_least(attempts: list[Attempt], backoffs: list[float]) -> bool:
    """Whether each retry came at least its backoff after the attempt before it."""
    waits = [later.at_s - earlier.at_s for earlier, later in pairwise(attempts)]
    if len(waits) != len(backoffs):
        return False
    return all(wait >= backoff for wait, backoff in zip(waits, backoffs, strict=True))


# --- with every default: today's timeouts and attempts --------------------------------


@pytest.mark.parametrize(
    ("component", "backend", "timeout_s"),
    [("embedder", "served", 30.0), ("reranker", "served", 30.0), ("llm", "api", 120.0)],
)
def test_with_every_default_a_component_makes_three_attempts_in_todays_timeout(
    settings_factory: SettingsFactory, component: str, backend: str, timeout_s: float
) -> None:
    endpoints = FlakyModelEndpoints({URLS[component]: ALWAYS})

    response = ask(settings_factory(**called_over_http(component, backend)), endpoints)

    assert response.status_code == 503, response.text
    assert response.json()["error"] == f"{component}_unavailable"
    attempts = endpoints.at(URLS[component])
    assert len(attempts) == 3  # the first attempt and QP_HTTP__RETRIES=2
    # QP_HTTP__TIMEOUT_S, or the LLM's own timeout in the tests' file.
    assert {attempt.timeout_s for attempt in attempts} == {timeout_s}
    assert waited_at_least(attempts, TODAYS_BACKOFFS)


# --- a component's own values ----------------------------------------------------------


@pytest.mark.parametrize("backend", ["served", "api"])
@pytest.mark.parametrize("component", ["embedder", "reranker", "llm"])
def test_a_component_on_served_or_api_has_its_own_timeout_and_retries(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    backend: str,
) -> None:
    upper = component.upper()
    set_environment(
        monkeypatch,
        {
            f"QP_COMPONENTS__{upper}__HTTP__TIMEOUT_S": "4.5",
            f"QP_COMPONENTS__{upper}__HTTP__RETRIES": "4",
            f"QP_COMPONENTS__{upper}__HTTP__BACKOFF_S": "0",
        },
    )
    endpoints = FlakyModelEndpoints({URLS[component]: ALWAYS})

    response = ask(settings_factory(**called_over_http(component, backend)), endpoints)

    assert response.json()["error"] == f"{component}_unavailable"
    attempts = endpoints.at(URLS[component])
    assert len(attempts) == 5  # the first attempt and its own 4 retries
    assert {attempt.timeout_s for attempt in attempts} == {4.5}


def test_the_components_of_one_app_are_each_called_with_their_own_values(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        components__vector_store__backend="fake",
        components__embedder__backend="served",
        components__embedder__served__url=URLS["embedder"],
        components__embedder__http={"timeout_s": 4.5},
        components__reranker__backend="api",
        components__reranker__api__url=URLS["reranker"],
        # More retries than the default's 2: only its own 3 rescue the call.
        components__reranker__http={"timeout_s": 6.5, "retries": 3, "backoff_s": 0},
        components__llm__api__url=URLS["llm"],
    )
    endpoints = FlakyModelEndpoints({URLS["reranker"]: 3})

    response = ask(settings, endpoints)

    assert response.status_code == 200, response.text
    seen = {
        name: (len(attempts), {attempt.timeout_s for attempt in attempts})
        for name, url in URLS.items()
        if (attempts := endpoints.at(url))
    }
    assert seen == {"embedder": (1, {4.5}), "reranker": (4, {6.5}), "llm": (1, {120.0})}


def test_a_component_waits_its_own_backoff_before_a_retry(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_environment(
        monkeypatch,
        {
            "QP_HTTP__BACKOFF_S": "0",  # every other component retries at once
            "QP_COMPONENTS__RERANKER__HTTP__RETRIES": "1",
            "QP_COMPONENTS__RERANKER__HTTP__BACKOFF_S": "0.25",
        },
    )
    endpoints = FlakyModelEndpoints({URLS["reranker"]: ALWAYS})

    ask(settings_factory(**called_over_http("reranker", "served")), endpoints)

    assert waited_at_least(endpoints.at(URLS["reranker"]), [0.25])


# --- one default for every component that sets none ------------------------------------


@pytest.mark.parametrize(
    ("component", "backend", "timeout_s"),
    [("embedder", "api", 5.0), ("reranker", "api", 5.0), ("llm", "served", 120.0)],
)
def test_the_default_set_once_holds_for_every_value_a_component_leaves_out(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    backend: str,
    timeout_s: float,
) -> None:
    set_environment(
        monkeypatch,
        {
            "QP_HTTP__TIMEOUT_S": "5",
            "QP_HTTP__RETRIES": "1",
            "QP_HTTP__BACKOFF_S": "0.25",
        },
    )
    endpoints = FlakyModelEndpoints({URLS[component]: ALWAYS})

    ask(settings_factory(**called_over_http(component, backend)), endpoints)

    attempts = endpoints.at(URLS[component])
    assert len(attempts) == 2
    # The LLM keeps the timeout of its own block in the tests' file.
    assert {attempt.timeout_s for attempt in attempts} == {timeout_s}
    assert waited_at_least(attempts, [0.25])
