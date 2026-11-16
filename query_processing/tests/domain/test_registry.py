"""``Registry``: named things of one sort, filled by a decorator when their module
is imported, and the import of every module of a package that fills one."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import pytest

from query_processing.domain.registry import Registry, import_package_modules

Greeting = Callable[[str], str]


def test_a_decorated_entry_is_returned_by_its_name() -> None:
    greetings: Registry[Greeting] = Registry("greeting")

    @greetings.register("hello")
    def hello(who: str) -> str:
        return f"hello {who}"

    assert greetings.get("hello") is hello
    assert greetings.get("hello")("you") == "hello you"


def test_a_name_registered_twice_is_refused_with_the_sort_and_the_name() -> None:
    greetings: Registry[Greeting] = Registry("greeting")
    greetings.register("hello")(str.upper)

    with pytest.raises(ValueError, match="registered") as refused:
        greetings.register("hello")(str.lower)

    assert str(refused.value) == (
        "greeting 'hello' is registered twice, by str.upper and then by str.lower "
        "(expected each name once)"
    )
    assert greetings.get("hello") is str.upper  # the first entry is kept


def test_an_unknown_name_names_the_sort_the_name_and_the_known_names() -> None:
    greetings: Registry[Greeting] = Registry("greeting")
    for name in ("salut", "hello"):
        greetings.register(name)(str.upper)

    with pytest.raises(ValueError, match="unknown") as refused:
        greetings.get("ciao")

    assert str(refused.value) == (
        "unknown greeting 'ciao' (expected one of: 'hello', 'salut')"
    )


def test_two_registries_of_one_sort_of_entry_stay_apart() -> None:
    first: Registry[Greeting] = Registry("greeting")
    second: Registry[Greeting] = Registry("greeting")
    first.register("hello")(str.upper)

    second.register("hello")(str.lower)  # no clash: another registry

    assert (first.get("hello"), second.get("hello")) == (str.upper, str.lower)


# --- a package that imports its own modules ------------------------------------------

REGISTRY_MODULE = """
from query_processing.domain.registry import Registry

COLOURS = Registry("colour")
"""
PACKAGE_INIT = """
from query_processing.domain.registry import import_package_modules

import_package_modules(__name__, __path__)
"""
ENTRY_MODULE = """
from .colours import COLOURS


@COLOURS.register("{name}")
def {name}() -> str:
    return "{name}"
"""


def write_package(root: Path, package: str, entries: tuple[str, ...]) -> None:
    """A package whose ``__init__`` imports its modules, one entry per module."""
    folder = root / package
    folder.mkdir()
    (folder / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8")
    (folder / "colours.py").write_text(REGISTRY_MODULE, encoding="utf-8")
    for name in entries:
        module = ENTRY_MODULE.format(name=name)
        (folder / f"{name}.py").write_text(module, encoding="utf-8")


def test_importing_the_package_registers_the_entry_of_every_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_package(tmp_path, "paints", ("red", "blue"))
    monkeypatch.syspath_prepend(str(tmp_path))

    colours = importlib.import_module("paints.colours").COLOURS

    assert [colours.get(name)() for name in ("red", "blue")] == ["red", "blue"]


def test_the_modules_of_a_package_are_imported_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_package(tmp_path, "inks", ("red",))
    monkeypatch.syspath_prepend(str(tmp_path))
    package = importlib.import_module("inks")

    import_package_modules(package.__name__, package.__path__)  # a second time

    colours = importlib.import_module("inks.colours").COLOURS
    assert colours.get("red")() == "red"  # not refused as registered twice
