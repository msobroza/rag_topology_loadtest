"""``qp-deploy tiers`` and ``qp-deploy environments``: ask Domino, then print a look.

Both only read. The internal route of revisions is asked only for the
environments named on the command line, once each.

Example:
    >>> show_tiers(http, ["cpu-16"], "details")  # prints that tier's fields as YAML
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..domino_lookups.environments import list_environments
from ..domino_lookups.hardware_tiers import list_hardware_tiers
from ..domino_lookups.internal_revisions import InternalRevisions
from .environments_listing import (
    environments_details,
    environments_table,
    revisions_table,
    shown_revisions,
)
from .listing_looks import Look
from .listing_names import only_named
from .tiers_listing import tiers_details, tiers_table


def show_tiers(http: httpx.Client, names: Sequence[str], look: Look) -> None:
    """Print Domino's hardware tiers, those named only when names are given.

    The look ``"list"`` prints the table for now.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        names: The tiers' names given on the command line; empty shows every one.
        look: How to print them.

    Raises:
        DominoDeployError: If the tiers cannot be read, or a name is no tier's.

    Example:
        >>> show_tiers(http, [], "table")
    """
    tiers = only_named(list_hardware_tiers(http), names, "hardware tier")
    print(tiers_details(tiers) if look == "details" else tiers_table(tiers), end="")


def show_environments(http: httpx.Client, names: Sequence[str], look: Look) -> None:
    """Print Domino's environments; one named is shown with all its revisions.

    The look ``"list"`` prints the table for now.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        names: The environments' names given on the command line; empty shows
            every one, with no revision but its documented three.
        look: How to print them.

    Raises:
        DominoDeployError: If the environments cannot be read, or a name is no
            environment's.

    Example:
        >>> show_environments(http, ["qp-rag"], "table")
    """
    environments = only_named(list_environments(http), names, "environment")
    internal = InternalRevisions(http)
    revisions = {e.id: shown_revisions(e, internal) for e in environments if names}
    if look == "details":
        print(environments_details(environments, revisions), end="")
    elif not names:
        print(environments_table(environments), end="")
    else:
        tables = (revisions_table(e, *revisions[e.id]) for e in environments)
        print("\n".join(tables), end="")
