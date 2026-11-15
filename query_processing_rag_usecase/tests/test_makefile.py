"""The use case's Makefile runs the toolkit's commands here, with no flag.

``make -n`` prints the commands without running them. Each command runs in this
directory, where it finds the use case's files and writes its output; the
toolkit's own files (its Locust file, its OVMS exporter, its compose file) are
named beside it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

TOOLS = "/tools/bin/"  # never run: printed only


def dry_run(directory: Path, target: str) -> list[str]:
    """The commands ``make -n <target>`` prints in ``directory``, one per line."""
    # A make running this suite would hand its own flags and variables down.
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    run = subprocess.run(
        ["make", "-n", target, f"BIN={TOOLS}"],
        cwd=directory,
        env={**env, "MAKELEVEL": "0"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return [line.strip() for line in run.stdout.splitlines()]


@pytest.mark.parametrize(
    ("target", "command"),
    [
        ("serve", "qp-serve"),
        ("ingest", "qp-ingest --synthetic 200"),
        ("calibrate", "qp-calibrate"),
        ("openvino-export", "qp-export-openvino"),
        ("deploy", "qp-deploy deploy all"),
        ("deploy-dry-run", "qp-deploy --dry-run deploy all"),
        ("status", "qp-deploy status"),
    ],
)
def test_each_command_runs_here_with_no_flag(
    use_case_dir: Path, target: str, command: str
) -> None:
    assert dry_run(use_case_dir, target) == [f"{TOOLS}{command}"]


def test_the_load_test_runs_the_toolkits_locust_file_and_writes_results_here(
    use_case_dir: Path,
) -> None:
    mkdir, *locust = dry_run(use_case_dir, "loadtest")

    assert mkdir == "mkdir -p results"
    command = " ".join(line.rstrip("\\") for line in locust)  # one, continued
    (locustfile,) = re.findall(r" -f (\S+) ", command)
    assert (use_case_dir / locustfile).resolve().is_file()
    assert "--csv results/monolith --html results/monolith.html" in command


def test_the_ovms_export_writes_into_this_directory(use_case_dir: Path) -> None:
    (command,) = dry_run(use_case_dir, "ovms-export")

    # No folder given: the exporter writes ovms/models in the working directory.
    assert command.endswith("/ovms/export_models.sh")
    assert (use_case_dir / command).resolve().is_file()


def test_the_local_infrastructure_serves_the_models_exported_here(
    use_case_dir: Path,
) -> None:
    (command,) = dry_run(use_case_dir, "infra-up")

    (compose_file,) = re.findall(r" -f (\S+) ", command)
    assert (use_case_dir / compose_file).resolve().is_file()
    assert " --project-directory . " in command
