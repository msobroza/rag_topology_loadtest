"""The router's pipeline file and its manifest: saved by training, checked at start-up.

A trained pipeline is a joblib file; beside it, with the same name and
``.json``, its manifest records what it was trained from (spec 9.3): the
file's hash, the scikit-learn version, the routes and a hash of their
utterances, the embedder's alias, query prefix and vector size, the training
values and the leave-one-out accuracy. It records the embedder's alias, not
its source id, which is a Dataset folder on Domino.

Start-up refuses a pipeline whose manifest differs from the settings, the
routes, the embedder or the installed scikit-learn, and a file whose bytes
differ from the hash: every message names the value found and the one
expected. The hash is checked before joblib reads the file.

Example:
    >>> pipeline = load_pipeline(settings, embedder_settings, routes)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict
from query_processing.extension_api import Route
from sklearn.pipeline import Pipeline

# What every refusal tells the user to do.
_RETRAIN = "train it again with qp-rag-train-router"


class PipelineSettings(Protocol):
    """The settings a pipeline is trained and loaded with (``KnnRouterSettings``)."""

    @property
    def pipeline_file(self) -> str:
        """The joblib file, its manifest beside it."""
        ...

    @property
    def n_neighbors(self) -> int:
        """The number of neighbours the classifier votes with."""
        ...

    @property
    def metric(self) -> str:
        """The distance the neighbours are found with."""
        ...


class PipelineManifest(BaseModel):
    """What a pipeline file was trained from, written beside it as JSON.

    Example:
        >>> manifest_of(Path("router_pipeline.joblib")).embedder_alias
        'Solon-embeddings-large-0.1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_file: str  # its name, without its folder
    pipeline_sha256: str
    scikit_learn_version: str
    routes: list[str]  # sorted, as the classifier's classes are
    utterances_sha256: str
    embedder_alias: str
    embedder_query_prefix: str
    vector_size: int
    n_neighbors: int
    metric: str
    leave_one_out_accuracy: float


class EmbedderIdentity(NamedTuple):
    """What a manifest records of the embedder: its alias and its query prefix."""

    alias: str
    query_prefix: str


def embedder_identity(embedder: BaseModel) -> EmbedderIdentity:
    """Read an embedder component's alias and query prefix from its settings.

    Args:
        embedder: The settings of the embedder the router reads.

    Returns:
        Its alias and its query prefix.

    Raises:
        ValueError: If the settings have no alias or no query prefix, as an
            embedder's have; the message shows them.

    Example:
        >>> embedder_identity(settings.components["router_embedder"]).alias
        'Solon-embeddings-large-0.1'
    """
    alias = getattr(embedder, "alias", None)
    query_prefix = getattr(embedder, "query_prefix", None)
    if not isinstance(alias, str) or not isinstance(query_prefix, str):
        raise ValueError(
            f"the router reads the component {embedder!r}; expected an embedder, "
            "whose settings have an alias and a query_prefix"
        )
    return EmbedderIdentity(alias, query_prefix)


def manifest_path(pipeline_file: Path) -> Path:
    """Return where a pipeline file's manifest is: the same name, with ``.json``.

    Args:
        pipeline_file: The joblib file.

    Returns:
        Its manifest's path.

    Example:
        >>> manifest_path(Path("router_pipeline.joblib"))
        PosixPath('router_pipeline.json')
    """
    return pipeline_file.with_suffix(".json")


def utterances_sha256(routes: Sequence[Route]) -> str:
    """Hash the routes' names and utterances, whatever order they are listed in.

    Args:
        routes: The routes a router chooses between.

    Returns:
        The SHA-256 of their names and utterances, as hex.

    Example:
        >>> utterances_sha256(settings.routes)[:8]
        '3f1c9a0b'
    """
    ordered = [[route.name, route.utterances] for route in _by_name(routes)]
    text = json.dumps(ordered, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_pipeline(
    pipeline: Pipeline, pipeline_file: Path, manifest: PipelineManifest
) -> PipelineManifest:
    """Write the pipeline with joblib, then its manifest with the file's hash.

    Args:
        pipeline: The fitted pipeline.
        pipeline_file: Where to write it.
        manifest: What it was trained from; its file name and hash are filled here.

    Returns:
        The manifest as written.

    Example:
        >>> save_pipeline(pipeline, Path("router_pipeline.joblib"), manifest)
    """
    joblib.dump(pipeline, pipeline_file)
    written = manifest.model_copy(
        update={
            "pipeline_file": pipeline_file.name,
            "pipeline_sha256": _sha256_of(pipeline_file),
        }
    )
    manifest_path(pipeline_file).write_text(
        written.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return written


def load_pipeline(
    settings: PipelineSettings, embedder: BaseModel, routes: Sequence[Route]
) -> Pipeline:
    """Load a pipeline file, once its manifest matches what the router is given.

    Args:
        settings: The router's pipeline file and training values.
        embedder: The settings of the embedder the router reads.
        routes: The routes the router chooses between.

    Returns:
        The fitted pipeline.

    Raises:
        ValueError: If the manifest is missing or records another value than
            the settings, the routes, the embedder or the installed
            scikit-learn give, or the file's bytes differ from its hash; the
            message names the value found and the value expected.

    Example:
        >>> load_pipeline(knn_settings, embedder_settings, routes).classes_
        array(['leans_on_history', 'self_contained'], dtype=object)
    """
    pipeline_file = Path(settings.pipeline_file)
    manifest = read_manifest(pipeline_file)
    expected = expected_manifest_values(settings, embedder, routes)
    for key, value in expected.items():
        _refuse_other(pipeline_file, key, getattr(manifest, key), value)
    _refuse_other(
        pipeline_file, "sha256", _sha256_of(pipeline_file), manifest.pipeline_sha256
    )
    return _pipeline_in(pipeline_file)


def read_manifest(pipeline_file: Path) -> PipelineManifest:
    """Read the manifest beside a pipeline file.

    Args:
        pipeline_file: The joblib file.

    Returns:
        Its manifest.

    Raises:
        ValueError: If the pipeline file or its manifest is missing, or the
            manifest is not one; the message names the file expected.

    Example:
        >>> read_manifest(Path("router_pipeline.joblib")).n_neighbors
        3
    """
    path = manifest_path(pipeline_file)
    if not pipeline_file.is_file() or not path.is_file():
        raise ValueError(
            f"no pipeline file {pipeline_file} with its manifest {path}; expected "
            f"both: {_RETRAIN}"
        )
    try:
        return PipelineManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(
            f"{path} is not a pipeline manifest ({exc}); expected the JSON "
            f"qp-rag-train-router writes: {_RETRAIN}"
        ) from exc


def expected_manifest_values(
    settings: PipelineSettings, embedder: BaseModel, routes: Sequence[Route]
) -> dict[str, object]:
    """Return what a manifest must record for a router given these settings.

    Args:
        settings: The router's pipeline file and training values.
        embedder: The settings of the embedder the router reads.
        routes: The routes the router chooses between.

    Returns:
        The manifest's keys checked at start-up, with the value each must have.

    Example:
        >>> expected_manifest_values(knn_settings, embedder, routes)["metric"]
        'cosine'
    """
    identity = embedder_identity(embedder)
    return {
        "pipeline_file": Path(settings.pipeline_file).name,
        "scikit_learn_version": sklearn.__version__,
        "routes": [route.name for route in _by_name(routes)],
        "utterances_sha256": utterances_sha256(routes),
        "embedder_alias": identity.alias,
        "embedder_query_prefix": identity.query_prefix,
        "n_neighbors": settings.n_neighbors,
        "metric": settings.metric,
    }


def _refuse_other(
    pipeline_file: Path, key: str, found: object, expected: object
) -> None:
    if found != expected:
        raise ValueError(
            f"the router's pipeline file {pipeline_file} has the {key} {found!r}; "
            f"expected {expected!r}: {_RETRAIN}"
        )


def _pipeline_in(pipeline_file: Path) -> Pipeline:
    pipeline = joblib.load(pipeline_file)  # its bytes are the ones hashed
    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            f"{pipeline_file} holds a {type(pipeline).__name__}; expected a "
            f"scikit-learn Pipeline: {_RETRAIN}"
        )
    return pipeline


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_name(routes: Sequence[Route]) -> list[Route]:
    return sorted(routes, key=lambda route: route.name)
