"""A component of a registered type on ``served`` with no URL is refused when the settings are read.

The four model types are refused so already. A router, or a type an extension
defines, on ``served`` is called at its model server's call route, the URL of
its ``served`` block: without one it would pass the settings and the start-up
checks, and be refused only once opened, after its embedder had loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from query_processing.config.routes_file import ROUTES_FILE_VAR
from query_processing.config.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import fake_router, write_routes_file
from shout_components import SHOUT_EXTENSION, write_shout_workflow


@pytest.fixture(autouse=True)
def routes_here(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A working directory whose ``routes.yaml`` holds the routes ``a`` and ``b``."""
    write_routes_file(tmp_path / "routes.yaml", ["a", "b"])
    monkeypatch.delenv(ROUTES_FILE_VAR, raising=False)
    monkeypatch.chdir(tmp_path)


def served_router(url: str | None) -> dict[str, object]:
    """A router on ``served``, with ``url`` in its served block when one is given."""
    router: dict[str, object] = {
        **fake_router("embedder", ["a", "b"]),
        "backend": "served",
    }
    return router if url is None else {**router, "served": {"url": url}}


def test_a_router_on_served_with_no_url_is_refused_naming_its_variables() -> None:
    with pytest.raises(ValidationError) as refused:
        WorkflowSettings(_env_file=None, components={"router": served_router(None)})  # type: ignore[call-arg]

    assert (
        "QP_COMPONENTS__ROUTER__BACKEND='served' needs "
        "QP_COMPONENTS__ROUTER__SERVED__URL, the full route of the endpoint "
        "(for example http://localhost:8888/call/router); got ''"
    ) in str(refused.value)


def test_a_router_on_served_at_a_url_is_read() -> None:
    url = "http://router-app:8888/call/router"

    settings = WorkflowSettings(_env_file=None, components={"router": served_router(url)})  # type: ignore[call-arg]

    assert settings.components["router"].served.url == url  # type: ignore[attr-defined]


def test_a_component_of_an_extensions_type_on_served_with_no_url_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shouter = {"backend": "served", "local": {"suffix": "!"}}
    path = write_shout_workflow(
        tmp_path / "workflow.yaml", [SHOUT_EXTENSION], {"shouter": shouter}
    )
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))

    with pytest.raises(ValidationError) as refused:
        WorkflowSettings(_env_file=None)  # type: ignore[call-arg]

    assert "QP_COMPONENTS__SHOUTER__SERVED__URL" in str(refused.value)
    assert "http://localhost:8888/call/shouter" in str(refused.value)
