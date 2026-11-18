"""The ports of the four components: who implements which, and what makes a port."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel, ValidationError

from query_processing.adapters.embedder.base_embedder import BaseEmbedder
from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.reranker.base_reranker import BaseReranker
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.adapters.vector_store.base_vector_store import (
    BaseLexicalSearch,
    BaseVectorStore,
)
from query_processing.adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from query_processing.adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.domain.components.base_component import COMPONENT_TYPES
from query_processing.domain.components.component_requests import (
    EmbedRequest,
    GenerateRequest,
    RerankRequest,
    SearchRequest,
)
from query_processing.domain.retrieval.passages import ScoredPassage
from workflow_fakes import (
    CapRecordingLLM,
    DenseOnlyStore,
    RecordingEmbedder,
    RecordingReranker,
    ScriptedStore,
    scored,
)


def test_elasticsearch_and_the_in_memory_store_rank_text_and_chroma_does_not() -> None:
    assert issubclass(ElasticsearchVectorStore, BaseLexicalSearch)
    assert isinstance(InMemoryVectorStore(), BaseLexicalSearch)
    # Open-source Chroma refuses ranked text search (Chroma Cloud only).
    assert not issubclass(ChromaVectorStore, BaseLexicalSearch)
    for store in (ElasticsearchVectorStore, InMemoryVectorStore, ChromaVectorStore):
        assert issubclass(store, BaseVectorStore)


def test_fakes_subclass_their_ports() -> None:
    assert isinstance(FakeEmbedder(), BaseEmbedder)
    assert isinstance(FakeReranker(), BaseReranker)
    assert isinstance(InMemoryVectorStore(), BaseVectorStore)
    assert isinstance(FakeLLM(), BaseLLM)
    assert not isinstance(FakeLLM(), BaseVectorStore)


def test_a_port_subclass_missing_a_method_cannot_be_instantiated() -> None:
    class HalfReranker(BaseReranker):
        """Implements ``rerank`` but forgets ``ready``."""

        async def rerank(
            self, query: str, passages: Sequence[ScoredPassage], top_n: int
        ) -> list[ScoredPassage]:
            return []

    with pytest.raises(TypeError, match=r"HalfReranker.*ready"):
        HalfReranker()  # type: ignore[abstract]


def test_a_class_with_the_right_methods_but_no_port_base_is_not_a_port() -> None:
    class DuckEmbedder:
        model_name = "duck"

        async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return []

        async def embed_query(self, text: str) -> list[float]:
            return []

        async def ready(self) -> tuple[bool, str]:
            return True, "duck"

    assert not isinstance(DuckEmbedder(), BaseEmbedder)


# --- the common API -------------------------------------------------------------


async def test_each_port_answers_ainvoke_from_its_own_method() -> None:
    embedder, reranker = RecordingEmbedder(), RecordingReranker()

    assert (await embedder.ainvoke(EmbedRequest(text="q"))).embedding == [1.0, 0.0]
    assert embedder.queries == ["q"]
    passages = [scored("a"), scored("b")]
    kept = await reranker.ainvoke(RerankRequest(query="q", passages=passages, top_k=1))
    assert [p.id for p in kept.passages] == ["a"]
    assert reranker.calls == [("q", 2, 1)]
    request = GenerateRequest(messages=[{"role": "user", "content": "hi"}])
    assert (await FakeLLM().ainvoke(request)).text == "[fake-llm] hi"


async def test_the_llm_answers_its_request_under_the_requests_output_schema() -> None:
    class Reply(BaseModel):
        text: str

    request = GenerateRequest(
        messages=[{"role": "user", "content": "hi"}], output_schema=Reply
    )

    assert (await FakeLLM().ainvoke(request)).text == '{"text":"[fake-llm] hi"}'


async def test_the_llm_is_given_the_cap_its_request_states() -> None:
    llm = CapRecordingLLM()
    messages = [{"role": "user", "content": "hi"}]

    await llm.ainvoke(GenerateRequest(messages=messages))
    await llm.ainvoke(GenerateRequest(messages=messages, max_tokens=32))

    assert llm.caps == [None, 32]


@pytest.mark.parametrize("cap", [0, -1])
def test_a_request_refuses_a_cap_below_one(cap: int) -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        GenerateRequest(messages=[], max_tokens=cap)


def test_the_four_ports_are_the_registered_component_types() -> None:
    # A test extension registers a type of its own once it is imported in the
    # same run (shout_extension), so the four are checked, not the count.
    assert {"embedder", "llm", "reranker", "vector_store"} <= set(
        COMPONENT_TYPES.names()
    )
    assert COMPONENT_TYPES.get("vector_store") is BaseVectorStore
    assert (BaseReranker.request_type, BaseReranker.component_type_name) == (
        RerankRequest,
        "reranker",
    )


@pytest.mark.parametrize(
    ("mode", "expected", "calls"),
    [
        ("dense", ["a", "b"], [("dense", [1.0, 0.0], 3)]),
        ("lexical", ["b", "c"], [("lexical", "q", 3)]),
        ("hybrid", ["b", "a", "c"], [("dense", [1.0, 0.0], 3), ("lexical", "q", 3)]),
    ],
)
async def test_the_vector_store_searches_by_its_retrieval_mode(
    mode: str, expected: list[str], calls: list[tuple[str, object, int]]
) -> None:
    store = ScriptedStore(dense=["a", "b"], lexical=["b", "c"])

    answer = await store.ainvoke(
        SearchRequest(
            embedding=[1.0, 0.0], text="q", top_k=3, retrieval_mode=mode, rrf_k=60
        )
    )

    assert [p.id for p in answer.passages] == expected
    assert store.calls == calls


async def test_a_store_that_ranks_no_text_is_refused_a_lexical_search() -> None:
    backend = "QP_COMPONENTS__VECTOR_STORE__BACKEND"
    request = SearchRequest(text="q", top_k=3, retrieval_mode="lexical")

    with pytest.raises(ValueError, match=rf"to elasticsearch \({backend} for the one"):
        await DenseOnlyStore(["a"]).ainvoke(request)


@pytest.mark.parametrize(
    ("fields", "refused"),
    [
        ({"retrieval_mode": "dense"}, r"retrieval_mode='dense' needs embedding"),
        ({"retrieval_mode": "lexical"}, r"retrieval_mode='lexical' needs text"),
        (
            {"retrieval_mode": "hybrid", "embedding": [1.0], "text": "q"},
            r"retrieval_mode='hybrid' needs rrf_k",
        ),
    ],
)
def test_a_search_request_refuses_what_its_retrieval_mode_cannot_use(
    fields: dict[str, object], refused: str
) -> None:
    with pytest.raises(ValueError, match=refused):
        SearchRequest.model_validate({"top_k": 3, **fields})


def test_a_request_is_frozen_and_refuses_an_unknown_key() -> None:
    request = EmbedRequest(text="q")

    with pytest.raises(ValueError, match="frozen"):
        request.text = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EmbedRequest.model_validate({"text": "q", "prefix": "query: "})
