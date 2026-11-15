"""The emulation's Makefile runs the toolkit's commands in the toolkit's directory.

Its Workspace targets hand the project's own work to the toolkit's Makefile
(``$(MAKE) -C $(TOOLKIT) <target>``) or run a command there: ``TOOLKIT`` must be
that directory, and each target it hands on one of that Makefile's. Read as text,
nothing is run.
"""

from __future__ import annotations

import re

from repo_paths import REPO_ROOT, TOOLKIT_ROOT

MAKEFILE = REPO_ROOT / "emulation" / "Makefile"


def handed_targets(makefile: str) -> list[str]:
    """The targets ``makefile`` runs with ``$(MAKE) -C $(TOOLKIT)``.

    Example::

        handed_targets("\\t$(MAKE) -C $(TOOLKIT) ingest\\n")   # ["ingest"]
    """
    return re.findall(r"\$\(MAKE\) -C \$\(TOOLKIT\) ([\w-]+)", makefile)


def makefile_targets(makefile: str) -> set[str]:
    """The names of the rules ``makefile`` defines, one per ``name:`` line."""
    return set(re.findall(r"^([\w-]+):", makefile, re.MULTILINE))


def test_the_readers_on_a_literal_makefile() -> None:
    makefile = (
        "ingest: ## the toolkit's\n"
        "\t$(MAKE) -C $(TOOLKIT) ingest\n"
        "loadtest:\n"
        "\t$(QUESTIONS) $(MAKE) -C $(TOOLKIT) loadtest-domino RUN_NAME=x\n"
        "\t$(MAKE) -C .. other\n"
    )
    assert handed_targets(makefile) == ["ingest", "loadtest-domino"]
    assert makefile_targets(makefile) == {"ingest", "loadtest"}


def test_its_toolkit_is_the_toolkits_directory() -> None:
    (toolkit,) = re.findall(
        r"^TOOLKIT = (\S+)$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert (MAKEFILE.parent / toolkit).resolve() == TOOLKIT_ROOT


def test_every_target_it_hands_on_is_one_of_the_toolkits() -> None:
    handed = handed_targets(MAKEFILE.read_text(encoding="utf-8"))
    toolkit = makefile_targets((TOOLKIT_ROOT / "Makefile").read_text(encoding="utf-8"))

    assert handed, "no target handed on: the reader is wrong"
    assert sorted(set(handed) - toolkit) == []


def test_it_runs_commands_only_in_the_toolkits_directory() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert re.findall(r"cd (\S+) &&", makefile) == ["$(TOOLKIT)", "$(TOOLKIT)"]
    assert "-C .." not in makefile
