"""The code-shape rules, read from the source with ``ast``: files of at most 500
lines, function bodies of at most 25 lines (docstring not counted), and at most 2
levels of ``if`` / ``for`` / ``while`` / ``try`` / ``with`` / ``match`` in a
function (a nested function or class is measured on its own)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from repo_paths import REPO_ROOT

ROOT = REPO_ROOT
MAX_FILE_LINES = 500
MAX_BODY_LINES = 25
MAX_NESTING = 2
BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
BLOCKS += (ast.Match,)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
Function = ast.FunctionDef | ast.AsyncFunctionDef


def python_files(*names: str) -> list[Path]:
    """Every ``.py`` file under (or at) ``names``, relative to the repository."""
    found: list[Path] = []
    for name in names:
        path = ROOT / name
        found += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    return found


def functions(source: str) -> Iterator[Function]:
    """Every function and method in ``source``, nested ones included."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def body_lines(function: Function) -> int:
    """Lines from the first statement after the docstring to the function's end."""
    body = function.body[1:] if ast.get_docstring(function) else function.body
    if not body or function.end_lineno is None:
        return 0
    return function.end_lineno - body[0].lineno + 1


def nesting(node: ast.AST, depth: int = 0) -> int:
    """How deep blocks nest under ``node``; an ``elif`` is not one level deeper."""
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            continue
        is_elif = isinstance(node, ast.If) and node.orelse == [child]
        opens = isinstance(child, BLOCKS) and not is_elif
        deepest = max(deepest, nesting(child, depth + opens))
    return deepest


def measured(*names: str) -> Iterator[tuple[str, Function]]:
    """``("file:line name", function)`` for every function under ``names``."""
    for path in python_files(*names):
        for function in functions(path.read_text(encoding="utf-8")):
            where = f"{path.relative_to(ROOT)}:{function.lineno} {function.name}"
            yield where, function


def test_the_measures_on_a_literal_source() -> None:
    source = '''def outer(x):
    """Doc."""
    if x:
        for item in x:
            pass
    elif x is None:
        with open(x) as f:
            pass

    def inner():
        while True:
            try:
                pass
            except ValueError:
                if x:
                    pass
'''
    outer, inner = functions(source)
    assert body_lines(outer) == 14
    assert nesting(outer) == 2
    assert nesting(inner) == 3


def test_no_file_is_longer_than_the_limit() -> None:
    too_long = [
        f"{path.relative_to(ROOT)}: {count} lines"
        for path in python_files("src", "tests", "locustfile.py")
        if (count := len(path.read_text(encoding="utf-8").splitlines()))
        > MAX_FILE_LINES
    ]
    assert not too_long, f"files over {MAX_FILE_LINES} lines: {too_long}"


def test_no_function_body_is_longer_than_the_limit() -> None:
    too_long = [
        f"{where}: {lines} lines"
        for where, function in measured("src", "locustfile.py")
        if (lines := body_lines(function)) > MAX_BODY_LINES
    ]
    assert not too_long, f"function bodies over {MAX_BODY_LINES} lines: {too_long}"


def test_no_function_nests_deeper_than_the_limit() -> None:
    too_deep = [
        f"{where}: {depth} levels"
        for where, function in measured("src")
        if (depth := nesting(function)) > MAX_NESTING
    ]
    assert not too_deep, f"functions nested over {MAX_NESTING} levels: {too_deep}"
