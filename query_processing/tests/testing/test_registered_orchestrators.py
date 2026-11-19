"""The orchestrators every test is run on: the registry's names, read once.

They are read when the tests are collected, so an orchestrator a test
registers for itself is never among them, and a library added under
``adapters/orchestrator/`` is with no test edited.
"""

from __future__ import annotations

import pytest

from query_processing.adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from query_processing.testing import registered_orchestrators
from workflow_fakes import RefusingOrchestrator, register_orchestrator

# As a test module reads them: when it is collected.
COLLECTED = registered_orchestrators()


def test_they_are_the_registry_s_names_with_the_reference_among_them() -> None:
    assert ORCHESTRATORS.names() == COLLECTED
    assert "langgraph" in COLLECTED  # the reference every other is held to


def test_an_orchestrator_a_test_registers_is_not_among_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_orchestrator(monkeypatch, "refusing", RefusingOrchestrator)

    assert "refusing" in ORCHESTRATORS.names()
    assert registered_orchestrators() == COLLECTED
