"""The orchestrator the settings choose: its class, and its check of the graph.

The workflow file's ``orchestrator`` names a class of ``ORCHESTRATORS``
(ADR 0010). ``check_registered_names`` has refused an unknown name before
either function here is called. The class is asked whether it runs the file's
graph once the toolkit's own checks accept it, before any component opens,
as a stage type is asked about its settings.

Example:
    >>> chosen_orchestrator(WorkflowSettings())
    <class 'LangGraphOrchestrator'>
"""

from __future__ import annotations

from ..adapters.orchestrator.base_orchestrator import ORCHESTRATORS, BaseOrchestrator
from ..application.graph_check import WorkflowGraphError
from ..config.workflow_settings import WorkflowSettings


def chosen_orchestrator(settings: WorkflowSettings) -> type[BaseOrchestrator]:
    """Return the orchestrator's class the settings name.

    Args:
        settings: The workflow settings, their ``orchestrator`` a registered name.

    Returns:
        The class registered under ``settings.orchestrator``.

    Example:
        >>> chosen_orchestrator(WorkflowSettings(orchestrator="langgraph"))
        <class 'LangGraphOrchestrator'>
    """
    return ORCHESTRATORS.get(settings.orchestrator)


def check_orchestrator_runs_graph(settings: WorkflowSettings) -> None:
    """Refuse a graph the chosen orchestrator would not run as LangGraph does.

    Args:
        settings: The workflow settings, their graph already accepted by the
            toolkit's checks.

    What the graph can run on instead is said here, once, for every
    orchestrator: the default, when it is not the one chosen and its own check
    accepts the graph. An adapter's refusal names no other orchestrator.

    Raises:
        WorkflowGraphError: The orchestrator's refusal, after its name:
            ``orchestrator 'mine': stage 'rerank' has ...; the default
            orchestrator 'langgraph' accepts this graph``.

    Example:
        >>> check_orchestrator_runs_graph(WorkflowSettings())  # langgraph: no raise
    """
    orchestrator = chosen_orchestrator(settings)
    try:
        orchestrator.check_graph_runs(settings.graph)
    except ValueError as exc:
        refusal = f"orchestrator {settings.orchestrator!r}: {exc}"
        raise WorkflowGraphError(refusal + _instead(settings)) from exc


def _instead(settings: WorkflowSettings) -> str:
    # The default from the settings' own field, so no name is written twice.
    default = WorkflowSettings.model_fields["orchestrator"].default
    if settings.orchestrator == default:
        return ""
    try:
        ORCHESTRATORS.get(default).check_graph_runs(settings.graph)
    except ValueError:
        return ""
    return f"; the default orchestrator {default!r} accepts this graph"
