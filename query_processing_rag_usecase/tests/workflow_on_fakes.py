"""This directory's workflow file, read as ``qp-serve`` reads it, on fake backends.

Each component of ``workflow.yaml`` is moved to its fake backend by the
variable its path names, as on a laptop: no weights, no pipeline file, no
store, no network. The settings are then refused as ``qp-serve`` refuses them
at start-up. A test runs from this directory first (conftest's
``use_case_dir``), then asks for the settings, or takes conftest's ``all_fake``.

Example::

    settings = all_fake_settings(monkeypatch, QP_GRAPH__STAGES__RERANK__TOP_K="5")
"""

from __future__ import annotations

import pytest
from query_processing.composition.composition_root import check_workflow_graph
from query_processing.composition.registered_names import check_registered_names
from query_processing.config.workflow_settings import WorkflowSettings

# The components of workflow.yaml, in its order.
COMPONENTS = (
    "embedder",
    "router_embedder",
    "router",
    "reranker",
    "llm",
    "vector_store",
)
FAKE_BACKENDS = {
    f"QP_COMPONENTS__{name.upper()}__BACKEND": "fake" for name in COMPONENTS
}
# The run label of the app on those backends: each component with its backend.
FAKE_RUN_LABEL = "+".join(f"{name}=fake" for name in COMPONENTS)


def settings_here() -> WorkflowSettings:
    """The settings qp-serve builds here, refused as it refuses them at start-up."""
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]
    check_registered_names(settings)
    check_workflow_graph(settings)
    return settings


def all_fake_settings(
    monkeypatch: pytest.MonkeyPatch, **variables: str
) -> WorkflowSettings:
    """The settings here with every component on fake, and ``variables`` set."""
    for name, value in {**FAKE_BACKENDS, **variables}.items():
        monkeypatch.setenv(name, value)
    return settings_here()
