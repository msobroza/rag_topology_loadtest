"""A registry of named things of one sort, filled by a decorator as modules import.

Each registry knows its entries by name and nothing else about them. An entry
registers itself, in its own module, with ``@REGISTRY.register(name)``; the
package that holds the entries imports all of its modules from its
``__init__.py`` with ``import_package_modules``, so a new entry is one new
module and no table is edited. The registry lives in the domain because every
layer that registers something may import the domain.

Example:
    >>> RERANK_API_FORMATS = Registry("rerank API format")
    >>> @RERANK_API_FORMATS.register("cohere")
    ... class CohereRerankFormat(BaseRerankApiFormat): ...
    >>> RERANK_API_FORMATS.get("cohere")
    <class 'CohereRerankFormat'>
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

_Entry = TypeVar("_Entry")


class Registry(Generic[_Entry]):
    """Named things of one sort: each registered once, each found by its name.

    Args:
        sort: What an entry is, named in every error: ``rerank API format``.

    Example:
        >>> greetings: Registry[Callable[[], str]] = Registry("greeting")
        >>> greetings.register("hello")(lambda: "hello")
        >>> greetings.get("hello")()
        'hello'
    """

    def __init__(self, sort: str) -> None:
        self._sort = sort
        self._entries: dict[str, _Entry] = {}

    def register(self, name: str) -> Callable[[_Entry], _Entry]:
        """Return the decorator that registers its target under ``name``.

        Args:
            name: The name the entry is found by.

        Returns:
            A decorator that registers what it decorates and returns it unchanged.

        Example:
            >>> @RERANK_API_FORMATS.register("cohere")
            ... class CohereRerankFormat(BaseRerankApiFormat): ...
        """

        def add(entry: _Entry) -> _Entry:
            if name in self._entries:
                raise ValueError(
                    f"{self._sort} {name!r} is registered twice "
                    "(expected each name once)"
                )
            self._entries[name] = entry
            return entry

        return add

    def get(self, name: str) -> _Entry:
        """Return the entry registered under ``name``.

        Args:
            name: The name asked for.

        Returns:
            The entry, as it was registered.

        Raises:
            ValueError: If nothing is registered under ``name``; the message
                names the sort, the name and every known name.

        Example:
            >>> RERANK_API_FORMATS.get("jina")
            ValueError: unknown rerank API format 'jina' (expected one of: 'cohere')
        """
        if name not in self._entries:
            known = ", ".join(repr(known_name) for known_name in sorted(self._entries))
            raise ValueError(
                f"unknown {self._sort} {name!r} (expected one of: {known})"
            )
        return self._entries[name]


def import_package_modules(package: str, path: Iterable[str]) -> None:
    """Import every module of a package, so that each one's registrations run.

    A package that holds the entries of a registry calls it from its
    ``__init__.py``: importing the package then registers every entry, and a
    module added to it is registered with no list to edit.

    Args:
        package: The package's dotted name: ``__name__`` in its ``__init__.py``.
        path: Where its modules are: ``__path__`` in its ``__init__.py``.

    Example:
        >>> import_package_modules(__name__, __path__)  # in a package's __init__
    """
    for module in pkgutil.iter_modules(path, prefix=f"{package}."):
        importlib.import_module(module.name)
