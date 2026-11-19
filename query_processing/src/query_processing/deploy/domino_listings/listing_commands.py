"""``qp-deploy tiers`` and ``qp-deploy environments``: ask Domino, then show a look.

Both only read. The look ``"list"`` opens the list of ``opening_list`` when
Textual is installed (the extra ``terminal``), and prints the table with a line
saying how to install it when not; no module but ``opening_list`` imports
Textual, and this one imports ``opening_list`` only inside the functions that
build or run the list. The internal route of revisions is asked once for each
environment named on the command line, or opened in the list.

Example:
    >>> show_tiers(http, ["cpu-16"], "details")  # prints that tier's fields as YAML
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Collection, Sequence
from functools import partial
from typing import TYPE_CHECKING

import httpx

from ..domino_lookups.environments import DominoEnvironment, list_environments
from ..domino_lookups.hardware_tiers import HardwareTier, list_hardware_tiers
from ..domino_lookups.internal_revisions import InternalRevisions
from ..domino_lookups.revision_sources import BaseRevisionSource
from .environments_listing import (
    environments_details,
    environments_table,
    revision_lines,
    revisions_table,
    shown_revisions,
)
from .listing_looks import Look
from .listing_names import only_named
from .tiers_listing import tiers_details, tiers_table

if TYPE_CHECKING:
    from .opening_list import OpeningItem

RunList = Callable[[str, Sequence["OpeningItem"]], None]
_NO_EXTRA = (
    'qp-deploy: the interactive list needs the toolkit\'s "terminal" extra '
    '(pip install -e "query_processing[terminal]" from the repository\'s root); '
    "here is the table"
)
_GAP = "   "  # between the parts of an item's label


def list_is_installed() -> bool:
    """Say whether Textual, which draws the list, is installed.

    Returns:
        True when the extra ``terminal`` (or ``dev``) is installed.

    Example:
        >>> list_is_installed()
        True
    """
    return importlib.util.find_spec("textual") is not None


def tier_items(
    tiers: Sequence[HardwareTier], opened: Collection[str]
) -> list[OpeningItem]:
    """Make one item of the list a tier, opening on every field Domino gave.

    Args:
        tiers: The tiers (see ``list_hardware_tiers``).
        opened: The names whose items start open (those given on the command line).

    Returns:
        The items, in Domino's order, each labelled with its name, size and id.

    Example:
        >>> tier_items(tiers, ())[1].label
        'cpu-16   16 cores, 64 GiB   id large-k8s'
    """
    from .opening_list import OpeningItem

    padded = _padded([tier.name for tier in tiers])
    return [
        OpeningItem(
            f"{name}{_GAP}{tier.size()}{_GAP}id {tier.id}",
            partial(_tier_lines, tier),
            starts_open=tier.name in opened,
            name=tier.name,
        )
        for tier, name in zip(tiers, padded, strict=True)
    ]


def environment_items(
    environments: Sequence[DominoEnvironment],
    internal: BaseRevisionSource,
    opened: Collection[str],
) -> list[OpeningItem]:
    """Make one item of the list an environment, its revisions asked when opened.

    Args:
        environments: The environments (see ``list_environments``).
        internal: The source of every revision (``InternalRevisions``), asked
            only when an item is first opened.
        opened: The names whose items start open (those given on the command line).

    Returns:
        The items, in Domino's order, each labelled with its name and id, and
        opening on its id, one line a revision, the note when the source has
        none, and its documented fields as YAML.

    Example:
        >>> environment_items(environments, InternalRevisions(http), ())[0].label
        'qp-rag   id env-1'
    """
    from .opening_list import OpeningItem

    padded = _padded([environment.name for environment in environments])
    return [
        OpeningItem(
            f"{name}{_GAP}id {environment.id}",
            partial(_environment_lines, environment, internal),
            starts_open=environment.name in opened,
            name=environment.name,
        )
        for environment, name in zip(environments, padded, strict=True)
    ]


def show_tiers(
    http: httpx.Client,
    names: Sequence[str],
    look: Look,
    run_list: RunList | None = None,
) -> None:
    """Show Domino's hardware tiers, those named only when names are given.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        names: The tiers' names given on the command line; empty shows every one.
            In the list, those named start open.
        look: How to show them.
        run_list: What runs the list; ``run_opening_list`` when not given.

    Raises:
        DominoDeployError: If the tiers cannot be read, or a name is no tier's.

    Example:
        >>> show_tiers(http, [], "table")
    """
    tiers = only_named(list_hardware_tiers(http), names, "hardware tier")
    if _list_runs_or_say_why(look):
        _runner(run_list)("Hardware tiers", tier_items(tiers, names))
        return
    print(tiers_details(tiers) if look == "details" else tiers_table(tiers), end="")


def show_environments(
    http: httpx.Client,
    names: Sequence[str],
    look: Look,
    run_list: RunList | None = None,
) -> None:
    """Show Domino's environments; one named is shown with all its revisions.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        names: The environments' names given on the command line; empty shows
            every one, with no revision but its documented three. In the list,
            those named start open.
        look: How to show them.
        run_list: What runs the list; ``run_opening_list`` when not given.

    Raises:
        DominoDeployError: If the environments cannot be read, or a name is no
            environment's.

    Example:
        >>> show_environments(http, ["qp-rag"], "table")
    """
    environments = only_named(list_environments(http), names, "environment")
    internal = InternalRevisions(http)
    if _list_runs_or_say_why(look):
        items = environment_items(environments, internal, names)
        _runner(run_list)("Environments", items)
        return
    revisions = {e.id: shown_revisions(e, internal) for e in environments if names}
    if look == "details":
        print(environments_details(environments, revisions), end="")
    elif not names:
        print(environments_table(environments), end="")
    else:
        tables = (revisions_table(e, *revisions[e.id]) for e in environments)
        print("\n".join(tables), end="")


# A terminal without the extra still gets the table: the person is told how to
# have the list instead, on the error output so that the table stays clean.
def _list_runs_or_say_why(look: Look) -> bool:
    if look != "list":
        return False
    if list_is_installed():
        return True
    print(_NO_EXTRA, file=sys.stderr)
    return False


def _runner(run_list: RunList | None) -> RunList:
    if run_list is not None:
        return run_list
    from .opening_list import run_opening_list

    return run_opening_list


def _padded(names: Sequence[str]) -> list[str]:
    width = max((len(name) for name in names), default=0)
    return [name.ljust(width) for name in names]


def _tier_lines(tier: HardwareTier) -> list[str]:
    return tiers_details([tier]).splitlines()


def _environment_lines(
    environment: DominoEnvironment, internal: BaseRevisionSource
) -> list[str]:
    revisions, note = shown_revisions(environment, internal)
    return [
        f"id {environment.id}",
        *revision_lines(environment, revisions),
        *([note] if note else []),
        *environments_details([environment], {}).splitlines(),
    ]
