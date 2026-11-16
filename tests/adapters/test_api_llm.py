"""``ApiLLM`` in the OpenAI chat format over ``httpx.MockTransport``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from model_api_fakes import HttpClients, ScriptedRoute, endpoint_client
from rag_load_test.adapters.llm.api.api_llm import ApiLLM
from rag_load_test.adapters.llm.api.openai_chat_format import OpenAIChatFormat
from rag_load_test.adapters.llm.base_llm import BaseLLM
from rag_load_test.domain.api_endpoint import ApiEndpoint
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.generation import Generation

URL = "https://gateway.test/v1/chat/completions"
MODEL = "gpt-test"
MESSAGES = [{"role": "user", "content": "Capital of France?"}]


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


def make(
    clients: HttpClients,
    *script: httpx.Response | Exception,
    temperature: float = 0.0,
    max_tokens: int = 0,
) -> tuple[ApiLLM, ScriptedRoute]:
    route = ScriptedRoute(*script)
    endpoint = ApiEndpoint(url=URL, api_format="openai", api_key="k1", model=MODEL)
    client = endpoint_client(clients.over(route), endpoint, component="llm")
    llm = ApiLLM(
        client, OpenAIChatFormat(), temperature=temperature, max_tokens=max_tokens
    )
    return llm, route


def completion(
    content: str | None, usage: dict[str, int] | None = None
) -> httpx.Response:
    choice = {"index": 0, "message": {"role": "assistant", "content": content}}
    body: dict[str, object] = {"id": "c1", "choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_returns_content_and_sends_model_and_temperature(
    clients: HttpClients, temperature: float
) -> None:
    llm, route = make(clients, completion("Paris."), temperature=temperature)
    assert isinstance(llm, BaseLLM)
    assert llm.model_name == MODEL
    assert await llm.generate(MESSAGES) == Generation("Paris.")
    [request] = route.requests
    assert (request.method, request.url) == ("POST", URL)
    assert request.json == {
        "model": MODEL,
        "messages": MESSAGES,
        "temperature": temperature,
    }
    assert request.authorization == "Bearer k1"


async def test_null_content_reads_as_empty(clients: HttpClients) -> None:
    llm, _ = make(clients, completion(None))
    assert (await llm.generate(MESSAGES)).text == ""


async def test_a_cap_is_sent_and_the_token_counts_come_back(
    clients: HttpClients,
) -> None:
    usage = {"prompt_tokens": 829, "completion_tokens": 128, "total_tokens": 957}
    llm, route = make(clients, completion("Paris.", usage), max_tokens=128)
    assert await llm.generate(MESSAGES) == Generation("Paris.", 829, 128)
    [request] = route.requests
    assert request.json["max_tokens"] == 128


async def test_a_cap_of_zero_is_left_out_of_the_body(clients: HttpClients) -> None:
    llm, route = make(clients, completion("Paris."), max_tokens=0)
    await llm.generate(MESSAGES)
    [request] = route.requests
    assert "max_tokens" not in request.json


@pytest.mark.parametrize(
    "body",
    [{"choices": []}, {"error": "x"}, {"choices": [{"text": "old API"}]}],
    ids=["no-choice", "no-choices", "no-message"],
)
async def test_malformed_answer_raises(
    clients: HttpClients, body: dict[str, object]
) -> None:
    llm, route = make(clients, httpx.Response(200, json=body))
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await llm.generate(MESSAGES)
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
    llm, route = make(clients, error)
    with pytest.raises(RagDependencyError) as info:
        await llm.generate(MESSAGES)
    assert (info.value.component, info.value.kind) == ("llm", kind)
    assert len(route.requests) == 3  # retried like any other call


async def test_a_rejected_key_fails_at_once(clients: HttpClients) -> None:
    llm, route = make(clients, httpx.Response(401, text="bad key"))
    with pytest.raises(RagDependencyError, match="HTTP 401"):
        await llm.generate(MESSAGES)
    assert len(route.requests) == 1
