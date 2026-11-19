"""The list that opens, in a terminal: items a person moves through and opens.

The only module of the toolkit that imports Textual, which sits in the optional
extra ``terminal``: an App never installs it, and no other module needs it to
load. The arrows move, the space bar opens or closes the item under the cursor,
``/`` narrows the items to those whose name holds the text typed, ``enter``
gives the keys back to the narrowed list, ``escape`` brings every item back,
and ``q`` quits.

Example:
    >>> item = OpeningItem("cpu-16", lambda: ["cores: 16"])
    >>> run_opening_list("Hardware tiers", [item])
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from textual.app import App
from textual.binding import Binding, BindingType
from textual.markup import escape
from textual.widget import Widget
from textual.widgets import Input, Static, Tree
from textual.widgets.tree import TreeNode

from ..apps_client import DominoDeployError

_KEYS = (
    "up/down move   space open/close   / filter   "
    "enter to the list   esc all items   q quit"
)


@dataclass(frozen=True)
class OpeningItem:
    """One item of the list: its line, and the lines it shows when open.

    ``lines`` is called once, when the item is first opened, so that what it asks
    of Domino is asked only for the items a person looks at. ``/`` matches the
    text typed against ``name``, the label when ``name`` is empty: a label that
    holds an id or a size matches on the name alone, as the design asks.

    Example:
        >>> OpeningItem("cpu-16   16 cores, 64 GiB", lambda: [], name="cpu-16")
    """

    label: str
    lines: Callable[[], Sequence[str]]
    starts_open: bool = False
    name: str = ""

    def matches(self, typed: str) -> bool:
        """Say whether the item's name holds the text typed, whatever the case.

        Args:
            typed: The filter's text.

        Returns:
            True when ``name`` (the label when empty) holds ``typed``.

        Example:
            >>> OpeningItem("cpu-16   id large-k8s", list, name="cpu-16").matches("K8S")
            False
        """
        return typed.casefold() in (self.name or self.label).casefold()


class OpeningList(App[None]):
    """A list of items, each opened and closed with the space bar.

    Args:
        title: What the items are (e.g. ``"Hardware tiers"``), the window's title.
        items: The items, in the order shown.

    Example:
        >>> OpeningList("Hardware tiers", items).run()
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("slash", "filter", "filter"),
        Binding("escape", "unfilter", "all items"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, title: str, items: Sequence[OpeningItem]) -> None:
        super().__init__()
        self.title = title
        self._items = list(items)
        self._lines: dict[int, Sequence[str]] = {}
        self._open: set[int] = {i for i, item in enumerate(items) if item.starts_open}

    def compose(self) -> Iterator[Widget]:
        """Lay out the line of keys, the hidden filter, and the tree of items.

        Yields:
            The line of keys, the filter, then the tree, its root hidden.
        """
        yield Static(_KEYS)
        yield Input(placeholder="filter")
        tree: Tree[int] = Tree(self.title)
        tree.show_root = False
        yield tree

    def on_mount(self) -> None:
        """Show every item, the filter hidden and the cursor on the first item."""
        self.query_one(Input).display = False
        self._show(range(len(self._items)))
        tree = self.query_one(Tree)
        tree.cursor_line = 0
        tree.focus()

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[int]) -> None:
        """Add an item's lines the first time it opens, and remember it open.

        Args:
            event: The tree's message that a node opened.
        """
        index = event.node.data
        if index is None:
            return
        self._open.add(index)
        if not event.node.children:
            self._add_lines(event.node, index)

    def on_tree_node_collapsed(self, event: Tree.NodeCollapsed[int]) -> None:
        """Remember an item closed, so that narrowing keeps it closed.

        Args:
            event: The tree's message that a node closed.
        """
        if event.node.data is not None:
            self._open.discard(event.node.data)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Narrow the items to those whose name holds the text, whatever the case.

        Args:
            event: The filter's message that its text changed.
        """
        self._show(i for i, item in enumerate(self._items) if item.matches(event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Give the keys back to the narrowed list, its cursor on the first item.

        Args:
            event: The filter's message that ``enter`` was pressed in it.
        """
        event.stop()
        tree = self.query_one(Tree)
        tree.cursor_line = 0
        tree.focus()

    def action_filter(self) -> None:
        """Show the filter and give it the keys."""
        filter_box = self.query_one(Input)
        filter_box.display = True
        filter_box.focus()

    def action_unfilter(self) -> None:
        """Clear and hide the filter, every item back, the keys to the tree."""
        filter_box = self.query_one(Input)
        filter_box.value = ""
        filter_box.display = False
        self._show(range(len(self._items)))
        self.query_one(Tree).focus()

    def _show(self, indexes: Iterable[int]) -> None:
        tree: Tree[int] = self.query_one(Tree)
        tree.clear()
        for index in indexes:
            node = tree.root.add(escape(self._items[index].label), data=index)
            if index in self._open:
                self._add_lines(node, index)
                node.expand()

    def _add_lines(self, node: TreeNode[int], index: int) -> None:
        for line in self._lines_of(index):
            node.add_leaf(escape(line))

    def _lines_of(self, index: int) -> Sequence[str]:
        if index not in self._lines:
            try:
                self._lines[index] = list(self._items[index].lines())
            except DominoDeployError as error:
                self._lines[index] = [str(error)]
        return self._lines[index]


def run_opening_list(title: str, items: Sequence[OpeningItem]) -> None:
    """Open the list in the terminal until the person quits it.

    Args:
        title: What the items are (e.g. ``"Environments"``).
        items: The items, in the order shown.

    Example:
        >>> items = environment_items(environments, source, [])
        >>> run_opening_list("Environments", items)
    """
    OpeningList(title, items).run()
