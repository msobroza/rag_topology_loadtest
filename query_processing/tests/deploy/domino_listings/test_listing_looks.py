"""Tests of the look a listing takes: the details, the table, or the list."""

from __future__ import annotations

import pytest

from query_processing.deploy.domino_listings.listing_looks import Look, chosen_look


@pytest.mark.parametrize(
    ("details", "table", "terminal", "look"),
    [
        (True, False, True, "details"),
        (True, False, False, "details"),
        (False, True, True, "table"),
        (False, True, False, "table"),
        (False, False, False, "table"),
        (False, False, True, "list"),
    ],
)
def test_the_look(details: bool, table: bool, terminal: bool, look: Look) -> None:
    assert chosen_look(details=details, table=table, terminal=terminal) == look
