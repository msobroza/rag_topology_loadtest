"""``OpenAIChatModel`` over a fake client, and what importing the adapters costs."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from rag_load_test.adapters.chat_model.openai_chat import OpenAIChatModel
from rag_load_test.domain.errors import RagDependencyError

HEAVY_MODULES = (
    "sentence_transformers",
    "torch",
    "elasticsearch",
    "openai",
    "joblib",
    "chromadb",
    "openvino",
    "optimum",
)


# --- OpenAIChatModel -------------------------------------------------------


class FakeChatCompletions:
    """``chat.completions.create`` double: returns ``outcome`` or raises it."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def fake_openai_client(outcome: Any) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions(outcome))
    )


def completion(content: str | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_openai_chat_returns_content_and_passes_model_and_temperature(
    temperature: float,
) -> None:
    client = fake_openai_client(completion("Paris."))
    messages = [{"role": "user", "content": "Capital of France?"}]
    chat_model = OpenAIChatModel(client, "gpt-test", temperature=temperature)

    assert await chat_model.generate(messages) == "Paris."
    assert client.chat.completions.calls == [
        {"model": "gpt-test", "messages": messages, "temperature": temperature}
    ]


def _openai_request() -> httpx.Request:
    return httpx.Request("POST", "http://x")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (openai.APITimeoutError(request=_openai_request()), "timeout"),
        (
            openai.APIStatusError(
                "boom",
                response=httpx.Response(500, request=_openai_request()),
                body=None,
            ),
            "unavailable",
        ),
    ],
    ids=["timeout", "status"],
)
async def test_openai_errors_map_to_llm_dependency_errors(
    error: Exception, kind: str
) -> None:
    client = fake_openai_client(error)
    with pytest.raises(RagDependencyError) as info:
        await OpenAIChatModel(client, "gpt-test", temperature=0.0).generate([])
    assert info.value.component == "llm"
    assert info.value.kind == kind
    assert info.value.target == "gpt-test"
    assert info.value.__cause__ is error


async def test_openai_chat_aclose_closes_the_sdk_client() -> None:
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    await OpenAIChatModel(
        SimpleNamespace(close=close), "gpt-test", temperature=0.0
    ).aclose()
    assert closed == [True]


# --- import cost -----------------------------------------------------------


def test_importing_adapters_leaves_heavy_libraries_unimported() -> None:
    # Fresh interpreter: this process may already have torch/openai loaded.
    code = (
        "import sys; "
        "import rag_load_test.adapters.local_models.sentence_transformer_models, "
        "rag_load_test.adapters.vector_store.elasticsearch_store, "
        "rag_load_test.adapters.chat_model.openai_chat, "
        "rag_load_test.adapters.vector_store.chroma_store, "
        "rag_load_test.adapters.hosted_models.openai_embedder, "
        "rag_load_test.adapters.hosted_models.rerank_api, "
        "rag_load_test.adapters.local_models.local_model, "
        "rag_load_test.adapters.local_models.openvino_export, "
        "rag_load_test.adapters.local_models.native_threads, "
        "rag_load_test.adapters.local_models.cpu_count; "
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    command = [sys.executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
