"""Build ``LoadTestSettings`` from a mapping of variables, nothing else.

Every variable the class reads is cleared first and no ``.env`` is read, so a
variable left in the shell or a stray ``.env`` cannot leak into a test.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from rag_load_test.loadtest.loadtest_settings import LoadTestSettings

LOADTEST_VARIABLES = tuple(
    LoadTestSettings.variable_of(field) for field in LoadTestSettings.model_fields
)


def clear_loadtest_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every variable ``LoadTestSettings`` reads, for this test only."""
    for name in LOADTEST_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def settings_from(
    monkeypatch: pytest.MonkeyPatch, variables: Mapping[str, str]
) -> LoadTestSettings:
    """``LoadTestSettings`` read from exactly ``variables`` (by variable name)."""
    clear_loadtest_variables(monkeypatch)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    return LoadTestSettings(_env_file=None)  # type: ignore[call-arg]
