"""Tests of the aligned table: padded columns, some aligned right, no trailing space."""

from __future__ import annotations

from query_processing.deploy.domino_listings.aligned_table import aligned_table


def test_the_columns_are_padded_and_a_column_aligned_right() -> None:
    table = aligned_table(
        ["NAME", "SIZE", "NOTE"],
        [["a", "1", "x"], ["longer", "", "yes"], ["b", "100", ""]],
        right={1},
    )
    assert table == (
        "NAME    SIZE  NOTE\n"
        "a          1  x\n"
        "longer        yes\n"
        "b        100\n"
    )
