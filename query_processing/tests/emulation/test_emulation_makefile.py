"""The emulation's Makefile runs the use case's commands in the use case's directory.

The emulation deploys the RAG use case. Its Workspace targets hand the use
case's own work to the use case's Makefile (``$(MAKE) -C $(USE_CASE) <target>``)
or run a command there, where it finds the use case's files: ``USE_CASE`` must
be that directory, and each target it hands on one of that Makefile's. Read as
text, nothing is run.
"""

from __future__ import annotations

import re

from emulation_terraform import USE_CASE_ROOT

from repo_paths import REPO_ROOT

MAKEFILE = REPO_ROOT / "emulation" / "Makefile"


def handed_targets(makefile: str) -> list[str]:
    """The targets ``makefile`` runs with ``$(MAKE) -C $(USE_CASE)``.

    Example::

        handed_targets("\\t$(MAKE) -C $(USE_CASE) ingest\\n")   # ["ingest"]
    """
    return re.findall(r"\$\(MAKE\) -C \$\(USE_CASE\) ([\w-]+)", makefile)


def makefile_targets(makefile: str) -> set[str]:
    """The names of the rules ``makefile`` defines, one per ``name:`` line."""
    return set(re.findall(r"^([\w-]+):", makefile, re.MULTILINE))


def test_the_readers_on_a_literal_makefile() -> None:
    makefile = (
        "ingest: ## the use case's\n"
        "\t$(MAKE) -C $(USE_CASE) ingest\n"
        "loadtest:\n"
        "\t$(QUESTIONS) $(MAKE) -C $(USE_CASE) loadtest-domino RUN_NAME=x\n"
        "\t$(MAKE) -C .. other\n"
    )
    assert handed_targets(makefile) == ["ingest", "loadtest-domino"]
    assert makefile_targets(makefile) == {"ingest", "loadtest"}


def test_its_use_case_is_the_use_cases_directory() -> None:
    (use_case,) = re.findall(
        r"^USE_CASE = (\S+)$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert (MAKEFILE.parent / use_case).resolve() == USE_CASE_ROOT


def test_every_target_it_hands_on_is_one_of_the_use_cases() -> None:
    handed = handed_targets(MAKEFILE.read_text(encoding="utf-8"))
    use_case = makefile_targets(
        (USE_CASE_ROOT / "Makefile").read_text(encoding="utf-8")
    )

    assert handed, "no target handed on: the reader is wrong"
    assert sorted(set(handed) - use_case) == []


def test_it_runs_commands_only_in_the_use_cases_directory() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert re.findall(r"cd (\S+) &&", makefile) == ["$(USE_CASE)"] * 4
    assert "-C .." not in makefile
    assert "TOOLKIT" not in makefile
