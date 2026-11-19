"""The orchestrator's check at start-up: asked after the toolkit's, its refusal named.

``check_workflow_graph`` runs the toolkit's checks of the graph, then asks the
chosen orchestrator's class whether it runs the graph. Both run on the settings
alone, before any component opens. The orchestrator that refuses is one the
test registers, and it leaves the registry as it found it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

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
        "this library does not run (expected plain edges only); the default "
        "orchestrator 'langgraph' accepts this graph"
    )

    with pytest.raises(WorkflowGraphError, match=f"^{re.escape(message)}$"):
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

    assert str(refused.value) == (
        "orchestrator 'pydantic_graph': 'rerank' has more than one way out at "
        "once; pydantic-graph ends a run when the first reaches END and cancels "
        "the others (expected one plain edge or one conditional edge out of each "
        "stage); the default orchestrator 'langgraph' accepts this graph"
    )


def with_rerank_named(name: str) -> dict[str, Any]:
    """The fixture graph, its stage ``rerank`` named ``name``."""
    graph = fixture_graph()
    stages = graph["stages"]
    graph["stages"] = {(name if s == "rerank" else s): v for s, v in stages.items()}
    graph["edges"] = [[name if n == "rerank" else n for n in c] for c in graph["edges"]]
    for edge in graph["conditional_edges"]:
        edge["source"] = name if edge["source"] == "rerank" else edge["source"]
    return graph


def test_a_refusal_of_the_default_offers_no_other_orchestrator(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(**FAKE_BACKENDS, graph=with_rerank_named("re:rank"))

    with pytest.raises(WorkflowGraphError) as refused:
        check_workflow_graph(settings)

    assert str(refused.value) == (
        "orchestrator 'langgraph': stage 're:rank' is no name LangGraph takes for "
        "a node: ':' is a reserved character and is not allowed in the node "
        "names. (expected a stage name LangGraph takes)"
    )


def test_a_refusal_offers_no_default_that_refuses_the_graph_too(
    settings_factory: SettingsFactory,
) -> None:
    graph = with_rerank_named("re:rank")
    graph["edges"].append(["re:rank", "END"])
    settings = settings_factory(
        **FAKE_BACKENDS, orchestrator="pydantic_graph", graph=graph
    )

    with pytest.raises(WorkflowGraphError) as refused:
        check_workflow_graph(settings)

    assert "'re:rank' has more than one way out at once" in str(refused.value)
    assert "langgraph" not in str(refused.value)


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
