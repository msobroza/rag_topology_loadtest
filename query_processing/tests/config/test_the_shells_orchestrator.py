"""The tests' session does not read the shell's ``QP_ORCHESTRATOR``.

The README has a person export the variable for ``make serve``; the gate run from
that shell then failed four tests of the default orchestrator (19 November 2026).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from query_processing.config.workflow_settings import WorkflowSettings
from repo_paths import TOOLKIT_ROOT

# The four that read the shell before the session took the variable out.
TESTS_OF_THE_DEFAULT = (
    "tests/config/test_workflow_settings.py"
    "::test_without_the_key_the_orchestrator_is_langgraph",
    "tests/entrypoints/start_up/test_orchestrator_at_start_up.py"
    "::test_the_start_up_log_line_names_the_orchestrator",
    "tests/composition/test_orchestrator_graph_check.py"
    "::test_a_refusal_of_the_default_offers_no_other_orchestrator",
    "tests/composition/test_orchestrator_graph_check.py"
    "::test_a_graph_pydantic_graph_refuses_is_a_graph_error_that_names_it",
)


def test_the_tests_of_the_default_pass_with_another_orchestrator_exported(
    tmp_path: Path,
) -> None:
    # A child session, for the variable is read as the session starts.
    env = {**os.environ, WorkflowSettings.variable_of("orchestrator"): "pydantic_graph"}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    command += [f"--basetemp={tmp_path / 'child'}", *TESTS_OF_THE_DEFAULT]

    run = subprocess.run(
        command,
        cwd=TOOLKIT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert run.returncode == 0, run.stdout[-3000:]
    assert f"{len(TESTS_OF_THE_DEFAULT)} passed" in run.stdout
