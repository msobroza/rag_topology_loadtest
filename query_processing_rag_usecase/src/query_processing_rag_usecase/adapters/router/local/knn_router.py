"""The kNN router: chooses a route with a trained scikit-learn pipeline.

The router's ``local`` backend. ``qp-rag-train-router`` fits the pipeline on
the vectors of its routes' utterances; at start-up the router loads it only
once its manifest matches the router's settings, routes and embedder (spec
9.3). A query's vector is routed to the route most of its nearest utterances
belong to, with the share of their votes as the confidence.

Example:
    >>> router = KnnRouter(load_pipeline(settings, embedder, routes))
    >>> router.invoke(RouteRequest(embedding=vector)).name
    'leans_on_history'
"""

from __future__ import annotations

from pydantic import BaseModel
from query_processing.extension_api import (
    BaseRouter,
    ComponentResources,
    RouteChoice,
    RouteRequest,
    RouterSettings,
)
from sklearn.pipeline import Pipeline

from ....config.knn_router_settings import KnnRouterSettings
from ....infrastructure.pipeline_artifact import load_pipeline


class KnnRouter(BaseRouter, backend="local", settings=KnnRouterSettings):
    """Chooses with a trained pipeline: a normalizer, then a k-nearest-neighbours vote.

    Args:
        pipeline: The fitted pipeline, its classes the route names.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

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
        return cls(load_pipeline(settings, embedder, routes))

    def invoke(self, request: RouteRequest) -> RouteChoice:
        """Choose the route most of the query's nearest utterances belong to.

        Args:
            request: The query's vector, from the embedder the router reads.

        Returns:
            The route, with the share of the neighbours' votes it got.

        Example:
            >>> router.invoke(RouteRequest(embedding=vector))
            RouteChoice(name='leans_on_history', confidence=0.67)
        """
        probabilities = self._pipeline.predict_proba([request.embedding])[0]
        best = int(probabilities.argmax())
        return RouteChoice(
            name=str(self._pipeline.classes_[best]),
            confidence=float(probabilities[best]),
        )
