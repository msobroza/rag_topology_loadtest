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
from query_processing.domain.components.component_requests import GenerateAnswer
from query_processing.domain.endpoints.api_endpoint import ApiEndpoint
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
    endpoint = ApiEndpoint(url=URL, api_format="openai", model=MODEL)
    client = endpoint_client(clients.over(route), endpoint, component="llm", key="k1")
    llm = ApiLLM(
        client,
        OpenAIChatFormat(),
        temperature=temperature,
        retries=retries,
        structured_output=STRUCTURED_OUTPUT_METHODS.get(structured_output)(),
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
    assert await llm.generate(MESSAGES) == GenerateAnswer(text="Paris.")
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


async def test_the_token_counts_come_back(clients: HttpClients) -> None:
    usage = {"prompt_tokens": 829, "completion_tokens": 128, "total_tokens": 957}
    llm, _ = make(clients, completion("Paris.", usage))
    assert await llm.generate(MESSAGES) == GenerateAnswer(
        text="Paris.", prompt_tokens=829, completion_tokens=128
    )


async def test_the_cap_reaches_the_endpoint_and_the_counts_come_back(
    clients: HttpClients,
) -> None:
    usage = {"prompt_tokens": 829, "completion_tokens": 128, "total_tokens": 957}
    llm, route = make(clients, completion("Paris.", usage))
    answer = await llm.generate(MESSAGES, max_tokens=128)
    assert answer == GenerateAnswer(
        text="Paris.", prompt_tokens=829, completion_tokens=128
    )
    [request] = route.requests
    assert request.json["max_tokens"] == 128


async def test_no_cap_is_left_out_of_the_body(clients: HttpClients) -> None:
    llm, route = make(clients, completion("Paris."))
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
    # A 403: a key refused with a 401 is tried again within the retries
    # (tests/entrypoints/qp_serve/test_qp_serve_key_sources.py).
    llm, route = make(clients, httpx.Response(403, text="bad key"))
    with pytest.raises(RagDependencyError, match="HTTP 403"):
        await llm.generate(MESSAGES)
    assert len(route.requests) == 1


# --- under an output schema -----------------------------------------------------


async def test_a_schema_is_sent_as_openais_response_format(
    clients: HttpClients,
) -> None:
    llm, route = make(clients, completion(' {"ok": true}\n'))

    answer = await llm.generate(MESSAGES, output_schema=Verdict)
    assert answer.text == '{"ok": true}'
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

    answer = await llm.generate(MESSAGES, output_schema=Verdict)
    assert answer.text == '{"ok": false}'
    assert len(route.requests) == 2
    assert route.requests[0].json == route.requests[1].json


async def test_a_cap_and_a_schema_reach_the_endpoint_in_each_request(
    clients: HttpClients,
) -> None:
    llm, route = make(clients, completion("not json"), completion('{"ok": true}'))

    await llm.generate(MESSAGES, output_schema=Verdict, max_tokens=64)
    assert len(route.requests) == 2
    for request in route.requests:
        assert request.json["max_tokens"] == 64
        assert request.json["response_format"]["type"] == "json_schema"


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
    assert json.loads(answer.text) == {"ok": True}
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

    assert (await llm.generate(MESSAGES)).text == "Paris."
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
    assert json.loads(answer.text) == {"ok": False}
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


# --- the token counts of an answer asked again -------------------------------------


async def test_an_answer_asked_again_reports_the_counts_of_every_attempt(
    clients: HttpClients,
) -> None:
    # The refused attempt was paid for: its tokens count.
    refused = completion("not json", {"prompt_tokens": 100, "completion_tokens": 7})
    matched = completion('{"ok": true}', {"prompt_tokens": 100, "completion_tokens": 5})
    llm, route = make(clients, refused, matched)

    answer = await llm.generate(MESSAGES, output_schema=Verdict)

    assert answer == GenerateAnswer(
        text='{"ok": true}', prompt_tokens=200, completion_tokens=12
    )
    assert len(route.requests) == 2


async def test_a_count_one_attempt_did_not_report_is_none(
    clients: HttpClients,
) -> None:
    refused = completion("not json", {"prompt_tokens": 100})
    matched = completion('{"ok": true}', {"prompt_tokens": 100, "completion_tokens": 5})
    llm, _ = make(clients, refused, matched)

    answer = await llm.generate(MESSAGES, output_schema=Verdict)

    assert (answer.prompt_tokens, answer.completion_tokens) == (200, None)
