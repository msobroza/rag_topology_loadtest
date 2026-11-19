"""The orchestrator's check at start-up: asked after the toolkit's, its refusal named.

``check_workflow_graph`` runs the toolkit's checks of the graph, then asks the
chosen orchestrator's class whether it runs the graph. Both run on the settings
alone, before any component opens. The orchestrator that refuses is one the
test registers, and it leaves the registry as it found it.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from query_processing.adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from query_processing.application.graph_check import WorkflowGraphError
from query_processing.composition.composition_root import check_workflow_graph
from query_processing.config.workflow_settings import WorkflowSettings
from rag_components import FAKE_BACKENDS
from rag_graphs import fixture_graph
from workflow_fakes import RefusingOrchestrator, register_orchestrator

SettingsFactory = Callable[..., WorkflowSettings]


@pytest.fixture
def refusing(monkeypatch: pytest.MonkeyPatch) -> str:
    """The name of an orchestrator that refuses a conditional edge, for this test."""
    register_orchestrator(monkeypatch, "refusing", RefusingOrchestrator)
    return "refusing"


def test_the_default_orchestrator_runs_the_fixture_graph(
    settings_factory: SettingsFactory,
) -> None:
    check_workflow_graph(settings_factory(**FAKE_BACKENDS))


def test_an_orchestrators_refusal_is_a_graph_error_that_names_it(
    settings_factory: SettingsFactory, refusing: str
) -> None:
    settings = settings_factory(**FAKE_BACKENDS, orchestrator=refusing)
    message = (
        "orchestrator 'refusing': stage 'rerank' has a conditional edge, which "
        "this library does not run (expected plain edges only)"
    )

    with pytest.raises(WorkflowGraphError, match=re.escape(message)):
        check_workflow_graph(settings)


def test_pydantic_graph_runs_the_fixture_graph(
    settings_factory: SettingsFactory,
) -> None:
    check_workflow_graph(
        settings_factory(**FAKE_BACKENDS, orchestrator="pydantic_graph")
    )


def test_a_graph_pydantic_graph_refuses_is_a_graph_error_that_names_it(
    settings_factory: SettingsFactory,
) -> None:
    graph = fixture_graph()
    # A plain edge beside rerank's conditional edge: the run would end at END
    # before generate answers.
    graph["edges"].append(["rerank", "END"])
    settings = settings_factory(
        **FAKE_BACKENDS, orchestrator="pydantic_graph", graph=graph
    )
    check_workflow_graph(settings_factory(**FAKE_BACKENDS, graph=graph))

    with pytest.raises(WorkflowGraphError) as refused:
        check_workflow_graph(settings)

    assert str(refused.value).startswith(
        "orchestrator 'pydantic_graph': 'rerank' has more than one way out at once;"
    )


def test_the_orchestrator_is_asked_only_about_a_graph_the_toolkit_accepts(
    settings_factory: SettingsFactory, refusing: str
) -> None:
    graph = fixture_graph()
    graph["stages"]["retrieve"]["type"] = "retrievee"
    settings = settings_factory(**FAKE_BACKENDS, orchestrator=refusing, graph=graph)

    with pytest.raises(WorkflowGraphError) as refused:
        check_workflow_graph(settings)

    assert "retrievee" in str(refused.value)
    assert "orchestrator" not in str(refused.value)


def test_an_orchestrator_registered_for_a_test_does_not_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = ORCHESTRATORS.names()

    with monkeypatch.context() as patched:
        register_orchestrator(patched, "refusing", RefusingOrchestrator)
        assert "refusing" in ORCHESTRATORS.names()

    assert ORCHESTRATORS.names() == before
