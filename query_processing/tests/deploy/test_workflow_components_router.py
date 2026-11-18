"""A workflow file that declares a router, as qp-deploy checks its overrides.

The workflow settings refuse a router that lists a route the routes file
lacks, so the deploy tool reads that file too, as a command finds it in the
use case's directory, where the workflow App runs: ``routes.yaml`` there, else
the one ``QP_ROUTES_FILE`` names. The file here is the fixture's with a fake
router that reads its embedder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from query_processing.config.files.routes_file import ROUTES_FILE_VAR
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.workflow_components import (
    WorkflowComponent,
    read_workflow_components,
    refuse_override_values,
    refuse_unknown_override_keys,
)
from rag_graphs import (
    fake_router,
    fixture_components,
    fixture_graph,
    write_routes_file,
    write_workflow_file,
)

# A site's value of the workflow file: any override makes the deploy tool
# check the whole settings, the router's routes included.
FEWER_RETRIES = {"http": {"retries": 0}}


@pytest.fixture
def routed_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A use case's directory: the fixture's file with a router, and its routes."""
    monkeypatch.delenv(ROUTES_FILE_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    write_routes_file(tmp_path / "routes.yaml", ["a", "b"])
    router = fake_router("embedder", ["a", "b"])
    components = {**fixture_components(), "router": router}
    return write_workflow_file(tmp_path / "workflow.yaml", fixture_graph(), components)


def test_a_file_with_a_router_passes_with_the_routes_file_beside_it(
    routed_workflow: Path,
) -> None:
    router = read_workflow_components(routed_workflow)["router"]

    assert router == WorkflowComponent(type="router", backend="fake")
    refuse_unknown_override_keys(FEWER_RETRIES, routed_workflow)
    refuse_override_values(FEWER_RETRIES, routed_workflow)


def test_an_override_to_a_route_the_routes_file_lacks_is_refused(
    routed_workflow: Path,
) -> None:
    overrides = {"components": {"router": {"routes": ["a", "c"]}}}

    refuse_unknown_override_keys(overrides, routed_workflow)
    with pytest.raises(DominoDeployError) as refused:
        refuse_override_values(overrides, routed_workflow)

    assert "router 'router' lists the route 'c'" in str(refused.value)
    assert "(expected one of its routes: 'a', 'b')" in str(refused.value)


def test_the_routes_file_a_variable_names_is_the_one_read(
    routed_workflow: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    named = write_routes_file(elsewhere / "routes.yaml", ["x", "y"])
    monkeypatch.setenv(ROUTES_FILE_VAR, str(named))

    with pytest.raises(DominoDeployError, match=r"route 'a'.*'x', 'y'"):
        refuse_override_values(FEWER_RETRIES, routed_workflow)


def test_an_override_to_an_embedder_the_file_lacks_is_refused(
    routed_workflow: Path,
) -> None:
    overrides = {"components": {"router": {"embedder": "reranker"}}}

    with pytest.raises(DominoDeployError, match="names the embedder 'reranker'"):
        refuse_override_values(overrides, routed_workflow)
