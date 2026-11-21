"""Every command reads a workflow file that declares the toolkit's own router.

The router is a component type of the toolkit that registers when its port is
imported. Wherever the settings are read, the toolkit's own registered types
are loaded before the components, as an extension's are, so a workflow file
that declares a router and names no extension is read by every command.

Each command runs in a fresh interpreter: this process has imported the
router's port already (``conftest.py`` imports the composition root), which
would hide a command that does not load it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from domino_deploy_helpers import WHEELS
from rag_graphs import (
    FIXTURE_PROMPTS,
    fake_router,
    fixture_components,
    fixture_graph,
    write_routes_file,
    write_workflow_file,
)
from repo_paths import TOOLKIT_ROOT

ROUTER_PORT = "query_processing.adapters.router.base_router"
# A deploy file whose override makes qp-deploy read the whole settings.
DEPLOY_FILE = {
    "app_name_prefix": "rag",
    "bootstrap_wheels": WHEELS,  # a deploy refuses one that does not exist
    "topology": "t",
    "topology_alias": {"t": {"embedder": "local", "reranker": "local"}},
    "defaults": {"environment_id": "env-1"},
    "overrides": {"http": {"retries": 0}},
}
# Each command's module and arguments: each reads the workflow file's components.
COMMANDS = {
    "qp-deploy": ("query_processing.deploy.qp_deploy", ["--dry-run", "deploy", "all"]),
    "qp-served-models": ("query_processing.entrypoints.qp_served_models", []),
    # A router has no model to export: refused once the settings are read.
    "qp-export-openvino": (
        "query_processing.entrypoints.qp_export_openvino",
        ["--only", "router"],
    ),
}


def use_case_dir(root: Path) -> Path:
    """A use case's directory: the fixture's file with a fake router, no extension."""
    (root / "query_processing" / "domino").mkdir(parents=True)
    (root / "query_processing" / "domino" / "app.sh").write_text("#!/bin/bash\n")
    use_case = root / "my_use_case"
    use_case.mkdir()
    components = {**fixture_components(), "router": fake_router("embedder", ["a", "b"])}
    write_workflow_file(use_case / "workflow.yaml", fixture_graph(), components)
    write_routes_file(use_case / "routes.yaml", ["a", "b"])
    (use_case / "prompts.yaml").write_text(FIXTURE_PROMPTS.read_text())
    (use_case / "deploy.yaml").write_text(yaml.safe_dump(DEPLOY_FILE))
    return use_case


def run_command(
    module: str, argv: list[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run a command's ``main`` in a fresh interpreter, no ``QP_*`` variable set."""
    code = (
        f"import sys\nfrom {module} import main\n"
        f"assert {ROUTER_PORT!r} not in sys.modules, 'imported with the command'\n"
        f"sys.exit(main({argv!r}))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("QP_")}
    env["PYTHONPATH"] = str(TOOLKIT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_command_reads_a_router_with_no_extension_named(
    command: str, tmp_path: Path
) -> None:
    module, argv = COMMANDS[command]

    result = run_command(module, argv, use_case_dir(tmp_path))

    output = result.stdout + result.stderr
    assert "unknown component type" not in output, output
    assert "imported with the command" not in output, output
    expected_exit = 2 if command == "qp-export-openvino" else 0
    assert result.returncode == expected_exit, output
