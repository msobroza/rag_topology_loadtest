"""The router's scikit-learn pipeline: a normalizer, then a k-nearest-neighbours vote.

Fitted on one vector per utterance, labelled with its route's name, and scored
by leave-one-out: each utterance routed by a pipeline fitted on the others.

Example:
    >>> fitted = fit_pipeline(vectors, labels, n_neighbors=3, metric="cosine")
    >>> fitted.pipeline.predict([vectors[0]])
    array(['self_contained'], dtype=object)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer


class FittedPipeline(NamedTuple):
    """A fitted pipeline and its leave-one-out accuracy."""

    pipeline: Pipeline
    leave_one_out_accuracy: float


def fit_pipeline(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    n_neighbors: int,
    metric: str,
) -> FittedPipeline:
    """Fit the router's pipeline on labelled vectors, and score it by leave-one-out.

    Args:
        vectors: One vector per utterance.
        labels: The name of each utterance's route, in the same order.
        n_neighbors: The neighbours that vote.
        metric: The distance they are found with.

    Returns:
        The pipeline fitted on every vector, and its leave-one-out accuracy.

    Raises:
        ValueError: If there are fewer than ``n_neighbors + 1`` utterances, so
            leave-one-out cannot leave ``n_neighbors`` to vote; the message
            names both numbers.

    Example:
        >>> fit_pipeline([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], ["a", "a", "b"],
        ...              n_neighbors=1, metric="cosine").leave_one_out_accuracy
        0.666...
    """
    if len(vectors) <= n_neighbors:
        raise ValueError(
            f"{len(vectors)} utterances to train on; expected more than "
            f"n_neighbors ({n_neighbors}), so that each left out has as many "
            "neighbours to vote"
        )
    rows, targets = [list(v) for v in vectors], list(labels)
    scores = cross_val_score(
        _unfitted(n_neighbors, metric), rows, targets, cv=LeaveOneOut()
    )
    pipeline = _unfitted(n_neighbors, metric).fit(rows, targets)
    return FittedPipeline(pipeline, float(scores.mean()))


def _unfitted(n_neighbors: int, metric: str) -> Pipeline:
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    return make_pipeline(Normalizer(), classifier)
