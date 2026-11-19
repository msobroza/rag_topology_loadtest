"""The use case's Makefile runs the toolkit's commands here, with no flag.

``make -n`` prints the commands without running them. Each command runs in this
directory, where it finds the use case's files and writes its output; the
toolkit's own files (its Locust file, its OVMS exporter, its compose file) are
named beside it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from query_processing.config.workflow_settings import WorkflowSettings

TOOLS = "/tools/bin/"  # never run: printed only


def dry_run(
    directory: Path, target: str, *variables: str, tools: Path | None = None
) -> list[str]:
    """The commands ``make -n <target>`` prints in ``directory``, one per line.

    ``variables`` are ``NAME=value`` arguments of make; ``tools``, when given,
    is the folder of stand-ins a recipe line that make runs even under ``-n``
    (one that calls ``$(MAKE)``) finds first, as its ``BIN`` and on its PATH.
    """
    # A make running this suite would hand its own flags and variables down.
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    if tools is not None:
        env["PATH"] = f"{tools}{os.pathsep}{env['PATH']}"
    bin_dir = f"{tools}/" if tools is not None else TOOLS
    run = subprocess.run(
        ["make", "-n", target, f"BIN={bin_dir}", *variables],
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
        ("ingest", "qp-ingest --synthetic 200 --embedder embedder"),
        ("train-router", "qp-rag-train-router"),
        ("calibrate", "qp-calibrate"),
        ("openvino-export", "qp-export-openvino"),
        ("deploy", "qp-deploy deploy all"),
        ("deploy-dry-run", "qp-deploy --dry-run deploy all"),
        ("status", "qp-deploy status"),
        ("tiers", "qp-deploy tiers"),
        ("environments", "qp-deploy environments"),
    ],
)
def test_each_command_runs_here_with_no_flag(
    use_case_dir: Path, target: str, command: str
) -> None:
    assert dry_run(use_case_dir, target) == [f"{TOOLS}{command}"]


@pytest.mark.parametrize("target", ["tiers", "environments"])
def test_a_listing_passes_its_args_to_qp_deploy(
    use_case_dir: Path, target: str
) -> None:
    command = dry_run(use_case_dir, target, "ARGS=qp-rag --details")
    assert command == [f"{TOOLS}qp-deploy {target} qp-rag --details"]


def test_train_router_never_allows_the_fake_embedder(use_case_dir: Path) -> None:
    # Only the tests train on the fake embedder: the command run here refuses it.
    (command,) = dry_run(use_case_dir, "train-router")

    assert "--allow-fake-embedder" not in command


def test_ingest_names_the_embedder_that_the_store_is_searched_with(
    use_case_dir: Path,
) -> None:
    (command,) = dry_run(use_case_dir, "ingest")
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]

    (named,) = re.findall(r" --embedder (\S+)", command)

    # The file declares two embedders, so qp-ingest asks which one fills the
    # store: the one whose vectors the retrieve stage searches with.
    assert named == settings.searched_embedder() == "embedder"
    with pytest.raises(ValueError, match="declares 2"):
        settings.ingestion_embedder()


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


# --- answer records -------------------------------------------------------------


def test_record_answers_writes_its_record_under_results_beside_the_load_tests(
    use_case_dir: Path,
) -> None:
    mkdir, command = dry_run(use_case_dir, "record-answers")

    assert mkdir == "mkdir -p results"
    assert command == (
        f"{TOOLS}qp-record-answers --host http://localhost:8888 --out results/monolith"
    )


def test_record_answers_domino_finds_the_host_and_the_token_as_the_load_test_does(
    use_case_dir: Path, tmp_path: Path
) -> None:
    # make -n runs a line that calls $(MAKE): qp-deploy and curl answer here.
    tools = tmp_path / "bin"
    tools.mkdir()
    stand_in(tools / "qp-deploy", "echo https://domino.test/workflow-app")
    stand_in(tools / "curl", "echo run-token")
    loadtest = recipe_of(use_case_dir, "loadtest-domino")

    lines = dry_run(use_case_dir, "record-answers-domino", tools=tools)

    assert recipe_of(use_case_dir, "record-answers-domino") == loadtest.replace(
        "$(MAKE) loadtest", "$(MAKE) record-answers"
    )
    assert (
        f"{tools}/qp-record-answers --host https://domino.test/workflow-app "
        "--out results/monolith" in lines
    )


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ((), "results/monolith results/split-all results/split-reranker"),
        (
            ("BASELINE=split-all",),
            "results/split-all results/monolith results/split-reranker",
        ),
    ],
    ids=["sorted", "baseline-first"],
)
def test_compare_answers_compares_every_record_under_results(
    use_case_dir: Path, tmp_path: Path, variables: tuple[str, ...], expected: str
) -> None:
    shutil.copy(use_case_dir / "Makefile", tmp_path / "Makefile")
    (tmp_path / "results").mkdir()
    for run in ("split-reranker", "monolith", "split-all"):
        (tmp_path / "results" / f"{run}_answers.json").write_text("{}")
    (tmp_path / "results" / "monolith_target.json").write_text("{}")  # no record

    (command,) = dry_run(tmp_path, "compare-answers", *variables)

    assert command == (
        f"{TOOLS}qp-compare-answers {expected} --out results/answers-comparison.md"
    )


def test_pin_answers_runs_the_pinned_records_test_to_make_the_records_again(
    use_case_dir: Path,
) -> None:
    # The test that compares the app with its pinned answer records, which the
    # option turns into the run that writes them.
    test_file = "tests/answer_records/test_pinned_answer_records.py"

    assert dry_run(use_case_dir, "pin-answers") == [
        f"{TOOLS}pytest {test_file} --pin-answers"
    ]
    assert (use_case_dir / test_file).is_file()


def stand_in(path: Path, line: str) -> None:
    """An executable at ``path`` that runs one shell line."""
    path.write_text(f"#!/bin/sh\n{line}\n")
    path.chmod(0o755)


def recipe_of(directory: Path, target: str) -> str:
    """The recipe lines of ``target`` in ``directory``'s Makefile, joined."""
    text = (directory / "Makefile").read_text(encoding="utf-8")
    (recipe,) = re.findall(rf"^{re.escape(target)}:.*\n((?:\t.*\n)+)", text, re.M)
    return str(recipe)
