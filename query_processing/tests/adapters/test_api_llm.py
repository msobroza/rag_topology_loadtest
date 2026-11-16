"""``ApiLLM`` in the OpenAI chat format over ``httpx.MockTransport``."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from model_api_fakes import HttpClients, ScriptedRoute, endpoint_client
from query_processing.adapters.llm.api.api_llm import ApiLLM
from query_processing.adapters.llm.api.base_structured_output_method import (
    STRUCTURED_OUTPUT_METHODS,
)
from query_processing.adapters.llm.api.format_instructions_method import (
    FormatInstructionsMethod,
)
from query_processing.adapters.llm.api.json_schema_method import JsonSchemaMethod
from query_processing.adapters.llm.api.openai_chat_format import OpenAIChatFormat
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.domain.api_endpoint import ApiEndpoint
from query_processing.domain.errors import RagDependencyError

URL = "https://gateway.test/v1/chat/completions"
MODEL = "gpt-test"
MESSAGES = [{"role": "user", "content": "Capital of France?"}]


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool


def make(
    clients: HttpClients,
    *script: httpx.Response | Exception,
    temperature: float = 0.0,
    retries: int = 2,
    structured_output: str = "json_schema",
) -> tuple[ApiLLM, ScriptedRoute]:
    route = ScriptedRoute(*script)
    endpoint = ApiEndpoint(url=URL, api_format="openai", api_key="k1", model=MODEL)
    client = endpoint_client(clients.over(route), endpoint, component="llm")
    llm = ApiLLM(
        client,
        OpenAIChatFormat(),
        temperature=temperature,
        retries=retries,
        structured_output=STRUCTURED_OUTPUT_METHODS.get(structured_output)(),
    )
    return llm, route


def completion(content: str | None) -> httpx.Response:
    choice = {"index": 0, "message": {"role": "assistant", "content": content}}
    return httpx.Response(200, json={"id": "c1", "choices": [choice]})


@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_returns_content_and_sends_model_and_temperature(
    clients: HttpClients, temperature: float
) -> None:
    llm, route = make(clients, completion("Paris."), temperature=temperature)
    assert isinstance(llm, BaseLLM)
    assert llm.model_name == MODEL
    assert await llm.generate(MESSAGES) == "Paris."
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
    assert await llm.generate(MESSAGES) == ""


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


# --- under an output schema -----------------------------------------------------


async def test_a_schema_is_sent_as_openais_response_format(
    clients: HttpClients,
) -> None:
    llm, route = make(clients, completion(' {"ok": true}\n'))

    assert await llm.generate(MESSAGES, output_schema=Verdict) == '{"ok": true}'
    [request] = route.requests
    assert request.json["messages"] == MESSAGES
    assert request.json["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Verdict",
            "schema": Verdict.model_json_schema(),
            "strict": True,
        },
    }


async def test_an_answer_that_does_not_match_is_asked_again(
    clients: HttpClients,
) -> None:
    fenced = '```json\n{"ok": true}\n```'
    llm, route = make(clients, completion(fenced), completion('{"ok": false}'))

    assert await llm.generate(MESSAGES, output_schema=Verdict) == '{"ok": false}'
    assert len(route.requests) == 2
    assert route.requests[0].json == route.requests[1].json


async def test_an_answer_that_never_matches_fails_showing_the_text(
    clients: HttpClients,
) -> None:
    answers = ['{"ok": "maybe"}', "not json", "still not"]
    llm, route = make(clients, *map(completion, answers), retries=2)

    with pytest.raises(RagDependencyError, match="still not") as info:
        await llm.generate(MESSAGES, output_schema=Verdict)
    assert (info.value.component, info.value.target) == ("llm", URL)
    assert "Verdict" in info.value.detail
    assert len(route.requests) == 3


async def test_with_no_retries_an_answer_that_does_not_match_is_asked_once(
    clients: HttpClients,
) -> None:
    llm, route = make(clients, completion("no"), retries=0)

    with pytest.raises(RagDependencyError, match="'no'"):
        await llm.generate(MESSAGES, output_schema=Verdict)
    assert len(route.requests) == 1


# --- an endpoint that imposes no schema ----------------------------------------------


async def test_format_instructions_writes_the_schema_in_the_messages(
    clients: HttpClients,
) -> None:
    fenced = '```json\n{"ok": true}\n```'
    llm, route = make(
        clients, completion(fenced), structured_output="format_instructions"
    )

    answer = await llm.generate(MESSAGES, output_schema=Verdict)
    assert json.loads(answer) == {"ok": True}
    [request] = route.requests
    assert "response_format" not in request.json
    sent = request.json["messages"]
    assert sent[:-1] == MESSAGES
    assert sent[-1]["role"] == "user"
    assert '"ok"' in sent[-1]["content"]


async def test_format_instructions_leave_free_text_alone(
    clients: HttpClients,
) -> None:
    llm, route = make(
        clients, completion("Paris."), structured_output="format_instructions"
    )

    assert await llm.generate(MESSAGES) == "Paris."
    [request] = route.requests
    assert request.json == {"model": MODEL, "messages": MESSAGES, "temperature": 0.0}


async def test_an_object_with_text_around_it_is_asked_again(
    clients: HttpClients,
) -> None:
    answers = ['Sure: {"ok": true}. Anything else?', '```\n{"ok": false}\n```']
    llm, route = make(
        clients, *map(completion, answers), structured_output="format_instructions"
    )

    answer = await llm.generate(MESSAGES, output_schema=Verdict)
    assert json.loads(answer) == {"ok": False}
    assert len(route.requests) == 2


@pytest.mark.parametrize(
    "answer",
    # A truncated object is not completed into one the schema would accept.
    ["I cannot answer that.", '```json\n{"ok": true\n```'],
    ids=["prose", "truncated"],
)
async def test_under_format_instructions_no_match_fails_showing_the_text(
    clients: HttpClients, answer: str
) -> None:
    llm, route = make(
        clients,
        completion(answer),
        retries=0,
        structured_output="format_instructions",
    )

    with pytest.raises(RagDependencyError) as info:
        await llm.generate(MESSAGES, output_schema=Verdict)
    assert repr(answer) in info.value.detail
    assert len(route.requests) == 1


def test_each_method_is_one_registered_class_and_another_is_refused() -> None:
    assert STRUCTURED_OUTPUT_METHODS.names() == ("format_instructions", "json_schema")
    assert STRUCTURED_OUTPUT_METHODS.get("json_schema") is JsonSchemaMethod
    found = STRUCTURED_OUTPUT_METHODS.get("format_instructions")
    assert found is FormatInstructionsMethod
    message = (
        "unknown structured output method 'grammar' "
        "(expected one of: 'format_instructions', 'json_schema')"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        STRUCTURED_OUTPUT_METHODS.get("grammar")
