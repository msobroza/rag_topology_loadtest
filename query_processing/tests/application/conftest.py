"""The orchestrator the run tests of the workflow and of its builder are built on.

A test that asks for ``orchestrator`` runs once on each registered one, read
when the tests are collected (``registered_orchestrators``); one that reaches
LangGraph's own graph is marked ``ON_LANGGRAPH_ALONE`` (``workflow_fakes``).
"""

from __future__ import annotations

import pytest

from query_processing.adapters.orchestrator.base_orchestrator import (
    ORCHESTRATORS,
    BaseOrchestrator,
)
from query_processing.testing import registered_orchestrators


@pytest.fixture(params=registered_orchestrators())
def orchestrator(request: pytest.FixtureRequest) -> type[BaseOrchestrator]:
    """The class of each registered orchestrator, one test run per name."""
    return ORCHESTRATORS.get(request.param)
