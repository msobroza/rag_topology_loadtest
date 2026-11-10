"""Shared fixtures. No model weights, no network, no Elasticsearch anywhere in this suite."""

from __future__ import annotations

import os

# locust gevent-patches the interpreter at import unless this is set; it must run
# before any ``import locust`` (the locustfile tests import it).
os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

from collections.abc import Callable  # noqa: E402

import pytest  # noqa: E402

from rag_load_test.corpus import chunk, ingest, synthetic_corpus  # noqa: E402
from rag_load_test.fakes import (  # noqa: E402
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.settings import RagSettings  # noqa: E402
from rag_load_test.workflow import RagDependencies  # noqa: E402

CORPUS_DOCS = 30  # 3 documents per synthetic topic


@pytest.fixture
def settings_factory() -> Callable[..., RagSettings]:
    """RagSettings without reading a stray ``.env``; keyword overrides allowed."""

    def make(**overrides: object) -> RagSettings:
        return RagSettings(_env_file=None, **overrides)  # type: ignore[call-arg]

    return make


@pytest.fixture
def fake_settings(settings_factory: Callable[..., RagSettings]) -> RagSettings:
    """Every backend set to ``fake``."""
    return settings_factory(
        embedder_backend="fake", reranker_backend="fake", llm_backend="fake"
    )


@pytest.fixture
async def fake_deps() -> RagDependencies:
    """Fake ports over an in-memory store populated with the synthetic corpus."""
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    passages = [p for doc in synthetic_corpus(CORPUS_DOCS) for p in chunk(doc)]
    await ingest(passages, embedder, store)
    return RagDependencies(embedder, store, FakeReranker(), FakeChatModel())
