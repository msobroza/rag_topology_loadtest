"""A router pipeline trained on the fake embedder in ``tmp_path``, and changes to it.

Example::

    trained = await train_on_fakes(tmp_path, monkeypatch)
    load_pipeline(*trained.after("n_neighbors"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic import BaseModel
from query_processing.extension_api import Route, WorkflowSettings
from router_workflow import settings_on_fakes

from query_processing_rag_usecase.config.knn_router_settings import KnnRouterSettings
from query_processing_rag_usecase.entrypoints.train_router import (
    router_parts,
    train_router,
)
from query_processing_rag_usecase.infrastructure.pipeline_artifact import (
    manifest_path,
)

LoadArguments = tuple[KnnRouterSettings, BaseModel, list[Route]]


class Trained(NamedTuple):
    """What a pipeline was trained with, and the workflow settings it came from."""

    settings: WorkflowSettings
    knn: KnnRouterSettings
    embedder: BaseModel
    routes: list[Route]

    def arguments(self) -> LoadArguments:
        """The arguments ``load_pipeline`` takes for this router, unchanged."""
        return self.knn, self.embedder, self.routes

    def after(self, change: str) -> LoadArguments:
        """The arguments ``load_pipeline`` takes once ``change`` was made.

        Raises:
            ValueError: If ``change`` is none of the changes listed here.
        """
        knn, embedder, routes = self.arguments()
        if change == "an utterance":
            first = routes[0]
            edited = [*first.utterances[:-1], first.utterances[-1] + " Merci."]
            routes = [first.model_copy(update={"utterances": edited}), *routes[1:]]
        elif change == "the embedder's alias":
            embedder = embedder.model_copy(update={"alias": "another-embedder"})
        elif change == "the query prefix":
            embedder = embedder.model_copy(update={"query_prefix": "passage: "})
        elif change == "n_neighbors":
            knn = knn.model_copy(update={"n_neighbors": 5})
        elif change == "the metric":
            knn = knn.model_copy(update={"metric": "euclidean"})
        elif change == "the scikit-learn version":
            _edit_manifest(Path(knn.pipeline_file), scikit_learn_version="1.0.0")
        elif change == "a byte of the file":
            _flip_last_byte(Path(knn.pipeline_file))
        else:
            raise ValueError(f"no change {change!r}")
        return knn, embedder, routes


async def train_on_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Trained:
    """Train the router of the router's workflow file; return what it read."""
    settings = settings_on_fakes(tmp_path, monkeypatch)
    await train_router(settings, "router", allow_fake_embedder=True)
    parts = router_parts(settings, "router")
    return Trained(settings, parts.knn, parts.embedder, parts.routes)


def _edit_manifest(pipeline_file: Path, **values: object) -> None:
    path = manifest_path(pipeline_file)
    manifest = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps({**manifest, **values}), "utf-8")


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
