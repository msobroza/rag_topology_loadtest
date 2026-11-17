"""The committed router pipeline still matches the use case's own files.

``router_pipeline.joblib`` and its manifest were trained on Solon's vectors on
a machine that has the model; this machine has none. The pipeline is read as
start-up reads it, with the settings of this directory's ``workflow.yaml`` and
``routes.yaml``: a change of an utterance, of the router's training values, of
its embedder's alias or query prefix, or of the scikit-learn pin, made without
a new training, fails here and says how to train it again. Each kind of change
is refused by the manifest's check, which ``test_pipeline_artifact.py`` runs
for each of them on a pipeline trained on the fake embedder; here a copy of
this directory with one utterance changed shows the refusal on the committed
pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from query_processing.extension_api import WorkflowSettings
from sklearn.pipeline import Pipeline

from query_processing_rag_usecase.entrypoints.train_router import router_parts
from query_processing_rag_usecase.infrastructure.pipeline_artifact import (
    load_pipeline,
    read_manifest,
)

# The files a copy of this directory needs to read the settings and the pipeline.
USE_CASE_FILES = (
    "workflow.yaml",
    "routes.yaml",
    "prompts.yaml",
    "router_pipeline.joblib",
    "router_pipeline.json",
)


def committed_pipeline(router_name: str = "router") -> Pipeline:
    """Load the router's pipeline file of the current directory, as start-up does.

    Raises:
        AssertionError: If start-up would refuse it; the message says to train
            it again on a machine that has the router's embedder's model.

    Example::

        committed_pipeline().classes_  # ['leans_on_history', 'self_contained']
    """
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]
    parts = router_parts(settings, router_name)
    try:
        return load_pipeline(parts.knn, parts.embedder, parts.routes)
    except ValueError as exc:
        model = getattr(parts.embedder, "source_id", parts.router.embedder)
        raise AssertionError(
            f"start-up refuses the committed router pipeline: {exc}. Run "
            f"`make train-router` on a machine that has the model {model} "
            "(it refuses the fake embedder), then commit router_pipeline.joblib "
            "and router_pipeline.json together"
        ) from exc


def test_the_committed_pipeline_matches_the_use_cases_files(
    use_case_dir: Path,
) -> None:
    del use_case_dir  # requested so the settings are read from this directory
    pipeline = committed_pipeline()

    assert list(pipeline.classes_) == ["leans_on_history", "self_contained"]


@pytest.mark.parametrize(
    "variables",
    [
        {"QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT": "int8"},
        {"QP_LOCAL_MODELS__RUNTIME": "torch"},
        {
            "QP_COMPONENTS__ROUTER_EMBEDDER__BACKEND": "served",
            "QP_COMPONENTS__ROUTER_EMBEDDER__SERVED__URL": "http://e.test/v3/embeddings",
        },
    ],
    ids=["int8-weights", "torch-runtime", "served-embedder"],
)
def test_the_committed_pipeline_is_accepted_however_its_embedder_runs(
    use_case_dir: Path, monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]
) -> None:
    # The manifest records the embedder's alias and query prefix, never how its
    # weights run: a served embedder's format is not visible to the workflow app.
    del use_case_dir  # requested so the settings are read from this directory
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    assert list(committed_pipeline().classes_) == ["leans_on_history", "self_contained"]


def test_the_committed_pipeline_routes_a_vector_of_its_size_to_a_route(
    use_case_dir: Path,
) -> None:
    size = read_manifest(use_case_dir / "router_pipeline.joblib").vector_size

    (route,) = committed_pipeline().predict([[1.0 / size**0.5] * size])

    assert size == 1024  # Solon's
    assert route in {"self_contained", "leans_on_history"}


def test_an_utterance_changed_without_a_training_is_refused(
    use_case_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in USE_CASE_FILES:
        shutil.copyfile(use_case_dir / name, tmp_path / name)
    routes = tmp_path / "routes.yaml"
    utterance = "Pourquoi est-ce obligatoire ?"
    text = routes.read_text("utf-8")
    assert utterance in text
    routes.write_text(text.replace(utterance, "Pourquoi ?"), "utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError, match=r"utterances_sha256.*make train-router"):
        committed_pipeline()
