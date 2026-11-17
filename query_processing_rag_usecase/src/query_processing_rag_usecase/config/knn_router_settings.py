"""The ``local`` block of a kNN router: its pipeline file and its training values.

No value has a default here: ``workflow.yaml`` pins them (spec 9.3). The
training command fits the pipeline with them, and start-up refuses a pipeline
file whose manifest records others.

Example:
    >>> KnnRouterSettings(pipeline_file="router_pipeline.joblib",
    ...                   n_neighbors=3, metric="cosine").n_neighbors
    3
"""

from __future__ import annotations

from pydantic import Field
from query_processing.extension_api import SettingsBlock


class KnnRouterSettings(SettingsBlock):
    """A kNN router's pipeline file, its number of neighbours and its metric."""

    # The joblib file qp-rag-train-router writes, its manifest beside it with
    # the same name and .json; relative to the use case's directory.
    pipeline_file: str = Field(min_length=1)
    n_neighbors: int = Field(ge=1)
    metric: str = Field(min_length=1)  # a metric of KNeighborsClassifier
