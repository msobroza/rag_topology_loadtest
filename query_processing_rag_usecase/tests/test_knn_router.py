"""The kNN router passes the toolkit's router tests, and opens at start-up once trained."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from query_processing.extension_api import (
    BaseRouter,
    CompositionRoot,
    RouteRequest,
    WorkflowSettings,
)
from query_processing.testing import RouterContract
from trained_router import Trained, train_on_fakes

from query_processing_rag_usecase.adapters.router.local.knn_router import KnnRouter
from query_processing_rag_usecase.entrypoints.train_router import embed_texts
from query_processing_rag_usecase.infrastructure.pipeline_artifact import (
    load_pipeline,
)

# A message of the route leans_on_history, word for word one of its utterances.
LEANING = "Et pour un véhicule loué ?"


class TestKnnRouter(RouterContract):
    trained: Trained
    vector: list[float]

    @pytest.fixture(autouse=True)
    async def pipeline_trained_on_the_fake_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.trained = await train_on_fakes(tmp_path, monkeypatch)
        settings = self.trained.settings
        self.vector = (await embed_texts(settings, "router_embedder", [LEANING]))[0]

    def make_component(self) -> BaseRouter:
        return KnnRouter(load_pipeline(*self.trained.arguments()))

    def make_request(self) -> RouteRequest:
        return RouteRequest(embedding=self.vector)

    def routes(self) -> Sequence[str]:
        return ("self_contained", "leans_on_history")

    async def test_an_utterance_is_routed_to_its_own_route(self) -> None:
        choice = await self.make_component().ainvoke(self.make_request())

        assert choice.name == "leans_on_history"
        assert choice.confidence >= 1 / 3  # at least one of its three neighbours


async def test_start_up_opens_the_router_on_local_once_it_is_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await train_on_fakes(tmp_path, monkeypatch)
    monkeypatch.setenv("QP_COMPONENTS__ROUTER__BACKEND", "local")
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]

    async with CompositionRoot(settings) as composition:
        opened = (await composition.workflow_dependencies()).components()

    assert isinstance(opened["router"], KnnRouter)


async def test_start_up_refuses_a_router_on_local_with_no_pipeline_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained = await train_on_fakes(tmp_path, monkeypatch)
    Path(trained.knn.pipeline_file).unlink()
    monkeypatch.setenv("QP_COMPONENTS__ROUTER__BACKEND", "local")
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]

    # The composition wraps the router's refusal in the error naming the component.
    with pytest.raises(
        Exception, match=r"router.*no pipeline file.*qp-rag-train-router"
    ):
        async with CompositionRoot(settings) as composition:
            await composition.workflow_dependencies()
