"""The project environment variables the fake hands every App it starts.

Domino keeps them per project and sets them in every run; the emulation reads
them from ``rag-fake-domino --project-env FILE``: ``KEY=VALUE`` lines, ``#``
comments and blank lines ignored, the value taken as is up to the line's end.
They hold secrets, so no message here ever carries a value.

Example:
    >>> load_project_env(Path("project.env"))
    {'RAG_ES_PASSWORD': '...'}
"""

from __future__ import annotations

import re
from pathlib import Path

_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_project_env(path: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines into the project's environment variables.

    Args:
        path: The file, e.g. ``/etc/fake-domino/project.env``.

    Returns:
        Each variable's value by name, in file order; a later line wins.

    Raises:
        ValueError: Naming ``path`` (and the line number for a malformed
            line, never its text) if the file cannot be read or a line is
            not ``KEY=VALUE`` with a shell variable name as ``KEY``.

    Example:
        >>> load_project_env(Path("project.env"))["RAG_ES_URL"]
        'http://10.10.0.5:9200'
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"project env file {path}: cannot be read: {exc}") from exc
    variables: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, equals, value = line.partition("=")
        if not equals or not _SHELL_NAME.fullmatch(name):
            raise ValueError(
                f"project env file {path}, line {number}: expected KEY=VALUE "
                "with KEY a shell variable name (the line is not shown: it "
                "may hold a secret)"
            )
        variables[name] = value
    return variables
