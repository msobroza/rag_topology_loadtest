"""BaseLLM over an HTTP endpoint that speaks a generation API format.

Under an output schema the schema goes to the endpoint the way the component's
structured output method says (OpenAI's ``response_format``, or an instruction
in the messages), and an answer the schema does not accept is asked again,
within the retries of the component's policy (spec 5).

Example:
    >>> llm = ApiLLM(endpoint_client, GENERATION_API_FORMATS.get("openai")(),
    ...              temperature=0.0, retries=2,
    ...              structured_output=JsonSchemaMethod())
    >>> answer = await llm.generate(messages, output_schema=Verdict)
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel, ValidationError

from ....domain.api_endpoint import ApiEndpoint
from ....domain.errors import RagDependencyError
from ....infrastructure.endpoint.api_endpoint_client import ApiEndpointClient
from ....infrastructure.endpoint.open_endpoint_client import (
    RunTokenSource,
    open_endpoint_client,
)
from ....infrastructure.http.http_policy import HttpPolicy
from ....infrastructure.http.json_post import describe_malformed
from ..base_llm import BaseLLM
from .base_generation_api_format import GENERATION_API_FORMATS, BaseGenerationApiFormat
from .base_structured_output_method import (
    STRUCTURED_OUTPUT_METHODS,
    BaseStructuredOutputMethod,
    PreparedRequest,
)


class ApiLLM(BaseLLM):
    """Answers conversations on one endpoint at a fixed temperature.

    Args:
        endpoint_client: The endpoint every request goes to (component ``llm``).
        api_format: Builds each request body and reads each answer.
        temperature: The sampling temperature sent with every request.
        retries: How many times an answer that does not match its output
            schema is asked again; the component's policy gives it.
        structured_output: How an output schema reaches the endpoint and how
            the answer's JSON object is read back.

    Example:
        >>> llm = ApiLLM(endpoint_client, OpenAIChatFormat(), temperature=0.0,
        ...              retries=2, structured_output=JsonSchemaMethod())
        >>> await llm.generate([{"role": "user", "content": "Hi"}])
        'Hello!'
    """

    def __init__(
        self,
        endpoint_client: ApiEndpointClient,
        api_format: BaseGenerationApiFormat,
        *,
        temperature: float,
        retries: int,
        structured_output: BaseStructuredOutputMethod,
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format
        self._temperature, self._retries = temperature, retries
        self._structured_output = structured_output
        self.model_name = endpoint_client.endpoint.model

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
    ) -> str:
        """Answer a conversation at the configured temperature.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            output_schema: The pydantic model the answer must follow, sent the
                way the structured output method says; None for free text.

        Returns:
            The model's answer text; "" when the answer carries no content.
            Under ``output_schema``, the JSON object alone, which it accepts.

        Raises:
            RagDependencyError: If the request fails (a timeout has kind
                ``timeout``), the answer is not the format's shape, or no
                answer within the retries matches ``output_schema`` (the
                message shows the last one).

        Example:
            >>> await llm.generate(messages, output_schema=Verdict)
            '{"ok": true}'
        """
        if output_schema is None:
            return await self._answer(PreparedRequest(messages, None))
        return await self._answer_matching(messages, output_schema)

    async def _answer(self, prepared: PreparedRequest) -> str:
        body = self._api_format.request_body(
            self.model_name,
            prepared.messages,
            self._temperature,
            prepared.response_format,
        )
        answer = await self._endpoint_client.post(body)
        try:
            return self._api_format.answer(answer)
        except ValidationError as exc:
            raise self._endpoint_client.malformed(exc) from exc

    async def _answer_matching(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> str:
        # Asked once, then once per retry, until an answer matches.
        method = self._structured_output
        prepared = method.prepare(messages, schema)
        for _ in range(self._retries):
            text = method.json_text(await self._answer(prepared))
            if _mismatch(schema, text) is None:
                return text
        text = method.json_text(await self._answer(prepared))
        mismatch = _mismatch(schema, text)
        if mismatch is not None:
            raise self._unmatched(schema, text, mismatch)
        return text

    def _unmatched(
        self, schema: type[BaseModel], text: str, mismatch: ValidationError
    ) -> RagDependencyError:
        attempts = self._retries + 1
        detail = (
            f"no answer in {attempts} attempt(s) matches the output schema "
            f"{schema.__name__} ({describe_malformed(mismatch)}; expected one JSON "
            f"object the schema accepts); the last answer: {text!r}"
        )
        return RagDependencyError(
            "llm", detail, target=self._endpoint_client.endpoint.url
        )


def _mismatch(schema: type[BaseModel], text: str) -> ValidationError | None:
    # Why the schema refuses the text, or None when it accepts it.
    try:
        schema.model_validate_json(text)
    except ValidationError as exc:
        return exc
    return None


def open_api_llm(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    tokens: RunTokenSource,
    policy: HttpPolicy,
    temperature: float,
    structured_output: str,
) -> ApiLLM:
    """Build the LLM of one generation endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The generation route, its API format, its key and its model.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call; its retries also
            bound how many times an answer off its schema is asked again.
        temperature: The sampling temperature sent with every request.
        structured_output: The name of the structured output method that
            takes an output schema to the endpoint.

    Returns:
        The LLM.

    Raises:
        ValueError: If ``endpoint.api_format`` names no generation API format,
            or ``structured_output`` no structured output method; the message
            lists the known names.

    Example:
        >>> open_api_llm(client, endpoint, tokens=tokens, policy=policy,
        ...              temperature=0.0, structured_output="json_schema")
    """
    api_format = GENERATION_API_FORMATS.get(endpoint.api_format)()
    method = STRUCTURED_OUTPUT_METHODS.get(structured_output)()
    endpoint_client = open_endpoint_client(
        client, endpoint, component="llm", tokens=tokens, policy=policy
    )
    return ApiLLM(
        endpoint_client,
        api_format,
        temperature=temperature,
        retries=policy.retries,
        structured_output=method,
    )
