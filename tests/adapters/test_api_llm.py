"""``ApiLLM`` in the OpenAI chat format over ``httpx.MockTransport``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from model_api_fakes import HttpClients, ScriptedRoute, endpoint_client
from rag_load_test.adapters.model_api.api_llm import ApiLLM
from rag_load_test.adapters.model_api.openai_chat_format import OpenAIChatFormat
from rag_load_test.domain.api_endpoint import ApiEndpoint
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.ports import BaseLLM

URL = "https://gateway.test/v1/chat/completions"
MODEL = "gpt-test"
MESSAGES = [{"role": "user", "content": "Capital of France?"}]


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


def make(
    clients: HttpClients, *script: httpx.Response | Exception, temperature: float = 0.0
) -> tuple[ApiLLM, ScriptedRoute]:
    route = ScriptedRoute(*script)
    endpoint = ApiEndpoint(url=URL, api_format="openai", api_key="k1", model=MODEL)
    client = endpoint_client(clients.over(route), endpoint, component="llm")
    chat_model = ApiLLM(client, OpenAIChatFormat(), temperature=temperature)
    return chat_model, route


def completion(content: str | None) -> httpx.Response:
    choice = {"index": 0, "message": {"role": "assistant", "content": content}}
    return httpx.Response(200, json={"id": "c1", "choices": [choice]})


@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_returns_content_and_sends_model_and_temperature(
    clients: HttpClients, temperature: float
) -> None:
    chat_model, route = make(clients, completion("Paris."), temperature=temperature)
    assert isinstance(chat_model, BaseLLM)
    assert chat_model.model_name == MODEL
    assert await chat_model.generate(MESSAGES) == "Paris."
    [request] = route.requests
    assert (request.method, request.url) == ("POST", URL)
    assert request.json == {
        "model": MODEL,
        "messages": MESSAGES,
        "temperature": temperature,
    }
    assert request.authorization == "Bearer k1"


async def test_null_content_reads_as_empty(clients: HttpClients) -> None:
    chat_model, _ = make(clients, completion(None))
    assert await chat_model.generate(MESSAGES) == ""


@pytest.mark.parametrize(
    "body",
    [{"choices": []}, {"error": "x"}, {"choices": [{"text": "old API"}]}],
    ids=["no-choice", "no-choices", "no-message"],
)
async def test_malformed_answer_raises(
    clients: HttpClients, body: dict[str, object]
) -> None:
    chat_model, route = make(clients, httpx.Response(200, json=body))
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await chat_model.generate(MESSAGES)
    assert (info.value.component, info.value.target) == ("llm", URL)
    assert len(route.requests) == 1


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.Response(500, text="boom"), "unavailable"),
    ],
    ids=["timeout", "5xx"],
)
async def test_failures_carry_component_llm_and_kind(
    clients: HttpClients, error: httpx.Response | Exception, kind: str
) -> None:
    chat_model, route = make(clients, error)
    with pytest.raises(RagDependencyError) as info:
        await chat_model.generate(MESSAGES)
    assert (info.value.component, info.value.kind) == ("llm", kind)
    assert len(route.requests) == 3  # retried like any other call


async def test_a_rejected_key_fails_at_once(clients: HttpClients) -> None:
    chat_model, route = make(clients, httpx.Response(401, text="bad key"))
    with pytest.raises(RagDependencyError, match="HTTP 401"):
        await chat_model.generate(MESSAGES)
    assert len(route.requests) == 1
