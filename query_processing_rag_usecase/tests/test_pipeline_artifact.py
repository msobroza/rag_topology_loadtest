"""The pipeline file and its manifest: accepted as trained, refused after any change."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from trained_router import Trained, train_on_fakes

from query_processing_rag_usecase.infrastructure.pipeline_artifact import (
    load_pipeline,
    manifest_path,
    utterances_sha256,
)


@pytest.fixture
async def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Trained:
    return await train_on_fakes(tmp_path, monkeypatch)


def test_the_pipeline_loads_as_it_was_trained(trained: Trained) -> None:
    pipeline = load_pipeline(*trained.arguments())

    assert list(pipeline.classes_) == ["leans_on_history", "self_contained"]


@pytest.mark.parametrize(
    "change",
    [
        "an utterance",
        "the embedder's alias",
        "the query prefix",
        "n_neighbors",
        "the metric",
        "the scikit-learn version",
        "a byte of the file",
    ],
)
def test_a_pipeline_is_refused_after_each_kind_of_change(
    change: str, trained: Trained
) -> None:
    with pytest.raises(ValueError, match="expected"):
        load_pipeline(*trained.after(change))


@pytest.mark.parametrize(
    ("change", "found", "expected"),
    [
        ("n_neighbors", "3", "5"),
        ("the embedder's alias", "'Solon-embeddings-large-0.1'", "'another-embedder'"),
        ("the query prefix", "'query : '", "'passage: '"),
        ("the scikit-learn version", "'1.0.0'", "'1.9.1'"),
    ],
)
def test_the_refusal_names_the_value_found_and_the_value_expected(
    change: str, found: str, expected: str, trained: Trained
) -> None:
    with pytest.raises(ValueError, match=re.escape(f"{found}; expected {expected}")):
        load_pipeline(*trained.after(change))


def test_a_pipeline_without_its_manifest_is_refused(trained: Trained) -> None:
    manifest_path(Path(trained.knn.pipeline_file)).unlink()

    with pytest.raises(ValueError, match=r"router_pipeline\.json.*qp-rag-train-router"):
        load_pipeline(*trained.arguments())


def test_the_utterances_hash_ignores_the_order_the_routes_are_listed_in(
    trained: Trained,
) -> None:
    routes = trained.routes

    assert utterances_sha256(routes) == utterances_sha256(list(reversed(routes)))
    assert utterances_sha256(routes) != utterances_sha256(routes[:1])
