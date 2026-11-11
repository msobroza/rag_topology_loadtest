"""``OpenAIEmbedder`` over a fake SDK client, and how ``open`` builds the real one."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from rag_load_test.adapters.hosted_models.openai_embedder import OpenAIEmbedder
from rag_load_test.adapters.shared.http_policy import HttpPolicy
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.ports import EmbedderPort

MODEL = "text-embedding-3-small"
POLICY = HttpPolicy(timeout_s=12.5, retries=3, backoff_s=0.0)


class FakeEmbeddings:
    """``embeddings.create`` double: one vector per text, rows in reverse order.

    The vector of text ``t`` is ``[float(len(t))]``; ``outcome`` (an exception)
    is raised instead, and ``drop`` rows are left off the end of each answer.
    """

    def __init__(self, outcome: BaseException | None = None, drop: int = 0) -> None:
        self.outcome, self.drop = outcome, drop
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.outcome is not None:
            raise self.outcome
        rows = [
            SimpleNamespace(index=i, embedding=[float(len(text))])
            for i, text in enumerate(kwargs["input"])
        ]
        kept = rows[: len(rows) - self.drop]
        return SimpleNamespace(data=kept[::-1])


class FakeOpenAIClient:
    """The slice of ``AsyncOpenAI`` the embedder uses: ``embeddings`` and ``close``."""

    def __init__(self, embeddings: FakeEmbeddings) -> None:
        self.embeddings = embeddings
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def embedder(fake: FakeEmbeddings, batch_size: int = 2) -> OpenAIEmbedder:
    return OpenAIEmbedder(FakeOpenAIClient(fake), MODEL, batch_size=batch_size)


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://openai.test/v1/embeddings")


# --- embed -----------------------------------------------------------------


def test_embedder_is_an_embedder_port() -> None:
    assert isinstance(embedder(FakeEmbeddings()), EmbedderPort)


async def test_embed_batches_and_keeps_input_order() -> None:
    fake = FakeEmbeddings()
    vectors = await embedder(fake, batch_size=2).embed(["a", "bb", "ccc", "dddd", "e"])
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]
    assert fake.calls == [
        {"model": MODEL, "input": ["a", "bb"]},
        {"model": MODEL, "input": ["ccc", "dddd"]},
        {"model": MODEL, "input": ["e"]},
    ]


async def test_embed_no_texts_makes_no_call() -> None:
    fake = FakeEmbeddings()
    assert await embedder(fake).embed([]) == []
    assert fake.calls == []


async def test_embed_row_count_mismatch_is_malformed() -> None:
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await embedder(FakeEmbeddings(drop=1)).embed(["a", "b"])
    assert (info.value.component, info.value.target) == ("embedder", MODEL)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (openai.APITimeoutError(request=_request()), "timeout"),
        (openai.APIConnectionError(request=_request()), "unavailable"),
        (
            openai.APIStatusError(
                "boom", response=httpx.Response(500, request=_request()), body=None
            ),
            "unavailable",
        ),
    ],
    ids=["timeout", "connection", "status"],
)
async def test_sdk_errors_map_to_embedder_dependency_errors(
    error: Exception, kind: str
) -> None:
    with pytest.raises(RagDependencyError) as info:
        await embedder(FakeEmbeddings(outcome=error)).embed(["a"])
    assert (info.value.component, info.value.kind) == ("embedder", kind)
    assert info.value.target == MODEL
    assert info.value.__cause__ is error


# --- ready, aclose, open ---------------------------------------------------


async def test_ready_makes_no_call() -> None:
    fake = FakeEmbeddings()
    assert await embedder(fake).ready() == (True, f"openai:{MODEL}")
    assert fake.calls == []


async def test_aclose_closes_the_sdk_client() -> None:
    client = FakeOpenAIClient(FakeEmbeddings())
    await OpenAIEmbedder(client, MODEL, batch_size=2).aclose()
    assert client.closed


async def test_open_passes_key_url_and_policy_to_the_sdk() -> None:
    built = OpenAIEmbedder.open(
        MODEL,
        api_key="sk-test",
        base_url="http://gateway.test/v1",
        batch_size=8,
        policy=POLICY,
    )
    client = built._client
    assert isinstance(client, openai.AsyncOpenAI)
    assert client.api_key == "sk-test"
    assert str(client.base_url) == "http://gateway.test/v1/"
    assert (client.timeout, client.max_retries) == (12.5, 3)
    assert built.model_name == f"openai:{MODEL}"
    await built.aclose()


async def test_open_with_blanks_leaves_the_sdk_to_its_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env.test/v1")
    built = OpenAIEmbedder.open(
        MODEL, api_key="", base_url="", batch_size=8, policy=POLICY
    )
    assert built._client.api_key == "sk-env"
    assert str(built._client.base_url) == "http://env.test/v1/"
    await built.aclose()


def test_importing_hosted_models_leaves_openai_unimported() -> None:
    # Fresh interpreter: this process has openai loaded already.
    code = (
        "import sys; "
        "import rag_load_test.adapters.hosted_models.openai_embedder, "
        "rag_load_test.adapters.hosted_models.rerank_api; "
        "print('openai' in sys.modules)"
    )
    command = [sys.executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
