"""A router's ``embedder`` names an embedder component, or the settings refuse it.

The router reads the vectors of the embedder its settings name. A name the
workflow file does not declare, or one of a component of another type, is
refused when the settings are read, naming the value found and the embedders
the file declares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from query_processing.config.router_settings import RouterSettings
from query_processing.config.routes_file import ROUTES_FILE_VAR
from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import fake_router, write_routes_file

SECOND_EMBEDDER = {"type": "embedder", "backend": "fake", "source_id": "org/second"}


@pytest.fixture(autouse=True)
def routes_here(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A working directory whose ``routes.yaml`` holds the routes ``a`` and ``b``."""
    write_routes_file(tmp_path / "routes.yaml", ["a", "b"])
    monkeypatch.delenv(ROUTES_FILE_VAR, raising=False)
    monkeypatch.chdir(tmp_path)


def settings_with_router(embedder: str, **components: Any) -> WorkflowSettings:
    """The fixture's settings and a fake router that reads ``embedder``."""
    router = fake_router(embedder, ["a", "b"])
    return WorkflowSettings(
        _env_file=None,  # type: ignore[call-arg]
        components={"router": router, **components},
    )


def test_a_router_that_names_a_component_of_another_type_is_refused() -> None:
    with pytest.raises(ValidationError) as refused:
        settings_with_router("reranker")

    assert (
        "router 'router' names the embedder 'reranker', which is no embedder "
        "component of the workflow file (its embedders: 'embedder')"
    ) in str(refused.value)


def test_a_router_that_names_no_component_is_refused_listing_each_embedder() -> None:
    with pytest.raises(ValidationError) as refused:
        settings_with_router("embeder", second_embedder=SECOND_EMBEDDER)

    assert (
        "names the embedder 'embeder', which is no embedder component of the "
        "workflow file (its embedders: 'embedder', 'second_embedder')"
    ) in str(refused.value)


def test_a_router_reads_an_embedder_declared_under_another_name() -> None:
    settings = settings_with_router("second_embedder", second_embedder=SECOND_EMBEDDER)

    router = settings.components["router"]
    assert isinstance(router, RouterSettings)
    assert router.embedder == "second_embedder"
