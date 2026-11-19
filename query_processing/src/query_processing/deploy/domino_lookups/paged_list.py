"""A list of Domino's documented API, read page after page.

Domino's lists take ``offset`` and ``limit`` and answer their items under one key,
with ``metadata.totalCount``. No ``limit`` is sent, so each page is as long as
Domino's default, and the next one starts where the items received end.

Example:
    >>> read_pages(http, "/api/hardwaretiers/v1/hardwaretiers", "hardwareTiers",
    ...            needs="listing the hardware tiers needs ViewHardwareTiers")
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn

import httpx

from ..apps_client import (
    DOMINO_API,
    DominoDeployError,
    domino_json,
    json_list,
    json_mapping,
    send_to_domino,
)


def read_pages(
    http: httpx.Client, path: str, key: str, *, needs: str
) -> list[Mapping[str, object]]:
    """Read every item of a Domino list.

    It asks ``offset`` 0, then the offset plus the items received, until it has
    read ``metadata.totalCount`` items or a page comes back empty (a list whose
    total is missing, or shrank while it was read, still ends). A page equal to
    the one before is refused: the list ignores ``offset``, and would never end.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).
        path: The list's path, e.g. ``"/api/hardwaretiers/v1/hardwaretiers"``.
        key: The key holding a page's items, e.g. ``"hardwareTiers"``.
        needs: What the list needs, said after Domino's answer to a 401 or 403.

    Returns:
        Every item, in Domino's order.

    Raises:
        DominoDeployError: If a call gets no answer, a status other than 200 (on
            401 or 403 followed by ``needs``), or a body that is not JSON, or if
            a page repeats the one before.

    Example:
        >>> read_pages(http, "/api/x", "items", needs="the list needs ViewX")
        [{'id': 'only'}]
    """
    items: list[Mapping[str, object]] = []
    previous: list[Mapping[str, object]] = []
    while True:
        page = _read_page(http, path, len(items), needs=needs)
        received = [json_mapping(item) for item in json_list(page.get(key))]
        if received and received == previous:
            _refuse_a_repeated_page(path, len(received), len(items))
        items += received
        previous = received
        total = json_mapping(page.get("metadata")).get("totalCount")
        if not received or (isinstance(total, int) and len(items) >= total):
            return items


def _refuse_a_repeated_page(path: str, count: int, offset: int) -> NoReturn:
    raise DominoDeployError(
        f"{DOMINO_API} GET {path} answered the same {count} items at offset "
        f"{offset} as at offset {offset - count}: the list repeats whatever the "
        "offset, so it cannot be read to its end"
    )


def _read_page(
    http: httpx.Client, path: str, offset: int, *, needs: str
) -> Mapping[str, object]:
    params = {"offset": str(offset)}
    response = send_to_domino(http, "GET", path, api=DOMINO_API, params=params)
    return domino_json(DOMINO_API, response, needs)
