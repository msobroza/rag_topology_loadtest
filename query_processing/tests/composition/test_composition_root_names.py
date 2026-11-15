"""``CompositionRoot`` opens components by name: what each command needs, nothing else."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.adapters.reranker.none.no_reranker import NoReranker
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from rag_components import FAKE_BACKENDS
from rag_graphs import rag_graph

SettingsFactory = Callable[..., WorkflowSettings]
UNREACHABLE_ES = "http://127.0.0.1:9"  # discard port: any connection would be refused


async def test_the_workflow_opens_the_components_its_stages_call_llm_first(
    settings_factory: SettingsFactory,
) -> None:
    # Lexical: no embed stage, so no embedder; Elasticsearch: nothing seeds it.
    settings = settings_factory(
        components__reranker__backend="none",
        components__llm__backend="fake",
        components__vector_store__elasticsearch__url=UNREACHABLE_ES,
        graph=rag_graph("lexical"),
    )
    async with CompositionRoot(settings) as composition:
        deps = await composition.workflow_dependencies()

    assert list(deps.by_name) == ["llm", "vector_store", "reranker"]


async def test_two_components_of_one_type_open_as_two(
    settings_factory: SettingsFactory,
) -> None:
    coarse = {"type": "reranker", "backend": "none", "source_id": "org/coarse"}
    # Merged into the fixture's graph: one more stage, calling the second reranker.
    rerank_coarse = {"type": "rerank", "call": "coarse", "top_k": 15}
    settings = settings_factory(
        **FAKE_BACKENDS,
        components={"coarse": coarse},
        graph={"stages": {"rerank_coarse": rerank_coarse}},
    )
    async with CompositionRoot(settings) as composition:
        deps = await composition.workflow_dependencies()

    assert isinstance(deps.by_name["reranker"], FakeReranker)
    assert isinstance(deps.by_name["coarse"], NoReranker)


async def test_a_component_is_opened_by_its_name_and_its_type(
    fake_settings: WorkflowSettings,
) -> None:
    async with CompositionRoot(fake_settings) as composition:
        with pytest.raises(ValueError, match="'reranker' is of type 'reranker', not"):
            await composition.local_embedder("reranker")
        with pytest.raises(ValueError, match="no component named 'gpu'"):
            await composition.local_reranker("gpu")
        with pytest.raises(ValueError, match="'llm' is of type 'llm': only an"):
            await composition.calibration("llm")


async def test_ingestion_embeds_with_the_embedder_named_when_there_are_two(
    settings_factory: SettingsFactory,
) -> None:
    second = {"type": "embedder", "backend": "fake", "source_id": "org/second"}
    settings = settings_factory(**FAKE_BACKENDS, components={"second": second})
    # Supplied: a fake store is seeded through the one embedder, and here are two.
    store = InMemoryVectorStore()
    async with CompositionRoot(settings, vector_store=store) as composition:
        with pytest.raises(ValueError, match=r"declares 2 .*'embedder', 'second'"):
            await composition.ingest_ports()
        embedder, _ = await composition.ingest_ports("second")
        deps = await composition.workflow_dependencies()

    assert embedder is not deps.by_name["embedder"]  # the embed stage's
    assert isinstance(embedder, FakeEmbedder)


async def test_ingestion_refuses_a_name_that_is_no_embedder(
    fake_settings: WorkflowSettings,
) -> None:
    async with CompositionRoot(fake_settings) as composition:
        with pytest.raises(ValueError, match="'reranker' is not an embedder"):
            await composition.ingest_ports("reranker")
