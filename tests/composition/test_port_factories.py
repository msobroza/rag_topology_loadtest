"""The port factories: one opener per Backend value, a supplied port used as given.

What each Backend's implementation does with the settings is tested through
``CompositionRoot`` in ``test_composition_root*.py``; here, the rule every
factory shares.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import get_args

import pytest

from rag_load_test.adapters.llm.base_llm import BaseLLM
from rag_load_test.adapters.llm.fake.fake_llm import FakeLLM
from rag_load_test.adapters.reranker.fake.fake_reranker import FakeReranker
from rag_load_test.composition.base_port_factory import BasePortFactory
from rag_load_test.composition.embedder_factory import EmbedderFactory
from rag_load_test.composition.http_resources import HttpResources
from rag_load_test.composition.llm_factory import LLMFactory
from rag_load_test.composition.local_model_resources import LocalModelResources
from rag_load_test.composition.reranker_factory import RerankerFactory
from rag_load_test.composition.resource_scope import Opener, ResourceScope
from rag_load_test.composition.vector_store_factory import VectorStoreFactory
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.backends import (
    EmbeddingBackend,
    GenerationBackend,
    RerankBackend,
    VectorStoreBackend,
)
from rag_load_test.infrastructure.local_models.cpu_count import ContainerCpus

BACKENDS = {
    "embedder": EmbeddingBackend,
    "reranker": RerankBackend,
    "llm": GenerationBackend,
    "vector_store": VectorStoreBackend,
}


def factories(
    settings: RagSettings, scope: ResourceScope
) -> dict[str, BasePortFactory]:
    """The four factories, wired as ``CompositionRoot`` wires them."""
    http = HttpResources(settings, scope)
    local_models = LocalModelResources(settings, scope, http, ContainerCpus())
    embedders = EmbedderFactory(settings, scope, http, local_models)
    return {
        "embedder": embedders,
        "reranker": RerankerFactory(settings, scope, http, local_models),
        "llm": LLMFactory(settings, scope, http),
        "vector_store": VectorStoreFactory(settings, scope, embedders.open),
    }


@pytest.mark.parametrize("port", sorted(BACKENDS))
def test_each_factory_has_one_opener_per_backend_value_of_its_port(
    port: str, fake_settings: RagSettings
) -> None:
    factory = factories(fake_settings, ResourceScope("test"))[port]

    assert factory.port == port
    assert set(factory.openers()) == set(get_args(BACKENDS[port]))
    assert factory.backend() == "fake"  # what fake_settings chooses everywhere


async def test_a_factory_opens_its_port_once_per_scope(
    fake_settings: RagSettings,
) -> None:
    async with ResourceScope("test") as scope:
        factory = factories(fake_settings, scope)["reranker"]
        reranker = await factory.open()

        assert isinstance(reranker, FakeReranker)
        assert await factory.open() is reranker


async def test_a_supplied_port_is_used_as_given_even_outside_a_scope(
    fake_settings: RagSettings,
) -> None:
    scope, supplied = ResourceScope("test"), FakeLLM()
    factory = LLMFactory(
        fake_settings, scope, HttpResources(fake_settings, scope), supplied
    )

    assert await factory.open() is supplied  # the scope was never entered


async def test_a_new_backend_is_one_opener_and_one_entry(
    fake_settings: RagSettings,
) -> None:
    class EchoFactory(BasePortFactory[BaseLLM]):
        """A factory with one Backend, written as a new one would be."""

        port = "llm"

        def backend(self) -> str:
            return "echo"

        def openers(self) -> Mapping[str, Opener[BaseLLM]]:
            return {"echo": self._echo}

        async def _echo(self) -> BaseLLM:
            return FakeLLM(latency_ms=1.0)

    async with ResourceScope("test") as scope:
        llm = await EchoFactory(fake_settings, scope, None).open()

    assert isinstance(llm, FakeLLM)
    assert llm.latency_ms == 1.0
