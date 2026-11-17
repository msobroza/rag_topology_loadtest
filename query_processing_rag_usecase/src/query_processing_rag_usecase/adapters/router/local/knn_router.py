"""The kNN router: chooses a route with a trained scikit-learn pipeline.

The router's ``local`` backend. ``qp-rag-train-router`` fits the pipeline on
the vectors of its routes' utterances; at start-up the router loads it only
once its manifest matches the router's settings, routes and embedder (spec
9.3). A query's vector is routed to the route most of its nearest utterances
belong to, with the share of their votes as the confidence.

Example:
    >>> router = KnnRouter(load_pipeline(settings, embedder, routes), vector_size=1024)
    >>> router.invoke(RouteRequest(embedding=vector)).name
    'leans_on_history'
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from query_processing.extension_api import (
    BaseRouter,
    ComponentResources,
    RagDependencyError,
    RouteChoice,
    RouteRequest,
    RouterSettings,
)
from sklearn.pipeline import Pipeline

from ....config.knn_router_settings import KnnRouterSettings
from ....infrastructure.pipeline_artifact import load_pipeline, read_manifest


class KnnRouter(BaseRouter, backend="local", settings=KnnRouterSettings):
    """Chooses with a trained pipeline: a normalizer, then a k-nearest-neighbours vote.

    Args:
        pipeline: The fitted pipeline, its classes the route names.
        vector_size: The length of the vectors it was trained on, from its
            manifest; a query's vector of another length is refused.
    """

    def __init__(self, pipeline: Pipeline, vector_size: int) -> None:
        self._pipeline = pipeline
        self._vector_size = vector_size

    @classmethod
    def from_settings(
        cls, settings: BaseModel | None, resources: ComponentResources
    ) -> KnnRouter:
        """Load the pipeline file, refused unless its manifest matches.

        Args:
            settings: The router's ``local`` block, read as ``KnnRouterSettings``.
            resources: The router's settings, its routes and its embedder's
                settings.

        Returns:
            The router, ready to answer.

        Raises:
            ValueError: If the block or the router's settings are not of their
                classes, or the pipeline file is refused; the message names
                the value found and the value expected.

        Example:
            >>> KnnRouter.from_settings(knn_settings, resources)
            <KnnRouter ...>
        """
        router = resources.type_settings
        if not isinstance(settings, KnnRouterSettings):
            raise ValueError(
                f"router {resources.name!r} has the local block {settings!r}; "
                "expected one with pipeline_file, n_neighbors and metric"
            )
        if not isinstance(router, RouterSettings):
            raise ValueError(
                f"router {resources.name!r} has the settings {router!r}; expected "
                "a router's, which name its embedder and its routes"
            )
        embedder = resources.component_settings(router.embedder)
        routes = list(resources.routes.values())
        pipeline = load_pipeline(settings, embedder, routes)
        return cls(pipeline, read_manifest(Path(settings.pipeline_file)).vector_size)

    def invoke(self, request: RouteRequest) -> RouteChoice:
        """Choose the route most of the query's nearest utterances belong to.

        Args:
            request: The query's vector, from the embedder the router reads.

        Returns:
            The route, with the share of the neighbours' votes it got.

        Raises:
            RagDependencyError: If the vector's length is not the one the
                pipeline was trained on: the embedder is not the one it was
                trained with. The message names both lengths.

        Example:
            >>> router.invoke(RouteRequest(embedding=vector))
            RouteChoice(name='leans_on_history', confidence=0.67)
        """
        if len(request.embedding) != self._vector_size:
            raise RagDependencyError(
                "router",
                f"the query's vector has {len(request.embedding)} numbers; expected "
                f"{self._vector_size}, the size the pipeline was trained on: train "
                "it again with qp-rag-train-router on the embedder the router reads",
            )
        probabilities = self._pipeline.predict_proba([request.embedding])[0]
        best = int(probabilities.argmax())
        return RouteChoice(
            name=str(self._pipeline.classes_[best]),
            confidence=float(probabilities[best]),
        )
