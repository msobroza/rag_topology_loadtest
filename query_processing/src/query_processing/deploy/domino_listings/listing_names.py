"""A listing's names: those given on the command line, and the name first in details.

Example:
    >>> [tier.id for tier in only_named(tiers, ["cpu-16"], "hardware tier")]
    ['large-k8s']
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar

import yaml

from ..apps_client import DominoDeployError


class _Named(Protocol):
    @property
    def name(self) -> str: ...


NamedItem = TypeVar("NamedItem", bound=_Named)


def only_named(
    items: Sequence[NamedItem], names: Sequence[str], kind: str
) -> list[NamedItem]:
    """Keep the items whose name is given, in Domino's order; every item for none.

    Two items of one name are both kept: Domino lets two share a name.

    Args:
        items: Domino's tiers or environments, in Domino's order.
        names: The names given on the command line; empty keeps every item.
        kind: What the items are, for the message (e.g. ``"hardware tier"``).

    Returns:
        The items named.

    Raises:
        DominoDeployError: If a name is no item's, with every name Domino has.

    Example:
        >>> [t.name for t in only_named(tiers, ["cpu-16", "cpu-8"], "hardware tier")]
        ['cpu-8', 'cpu-16']
    """
    if not names:
        return list(items)
    known = [item.name for item in items]
    for name in names:
        if name not in known:
            raise DominoDeployError(
                f"no {kind} of Domino is named {name!r}; it has: "
                f"{', '.join(known) or 'none'}"
            )
    return [item for item in items if item.name in names]


def details_yaml(items: Sequence[Mapping[str, object]]) -> str:
    """Write the items as YAML, each as given, its ``name`` moved first.

    No field is picked by hand: a field of another Domino version shows too.

    Args:
        items: Each item's fields as Domino gave them (plus what a listing adds).

    Returns:
        A YAML list, in the order of ``items``.

    Example:
        >>> print(details_yaml([{"id": "x", "name": "odd"}]), end="")
        - name: odd
          id: x
    """
    shown = [_name_first(item) for item in items]
    return yaml.safe_dump(shown, sort_keys=False, allow_unicode=True)


def _name_first(fields: Mapping[str, object]) -> dict[str, object]:
    if "name" not in fields:
        return dict(fields)
    return {"name": fields["name"]} | {k: v for k, v in fields.items() if k != "name"}
