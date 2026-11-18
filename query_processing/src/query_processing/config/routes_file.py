"""The routes file: where it is found, and the settings source that reads it.

A use case states its routes in ``routes.yaml``, beside its workflow file
(spec 8): a catalogue under the top key ``routes``, each route with a name, a
description and its utterances (``domain.components.routes``). Several routers can each
list some of them. It feeds the top-level setting ``routes`` of
``WorkflowSettings``, below the environment. A command finds it as it finds
the prompts file: the one ``QP_ROUTES_FILE`` names, else ``routes.yaml`` in
the working directory. A use case with neither has no routes, which is enough
when it declares no router. The workflow file holds no ``routes`` key.

Example:
    >>> source = routes_file_source(WorkflowSettings, routes_file())
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, InitSettingsSource

ROUTES_FILE = "routes.yaml"  # in the working directory
# Read from the environment only, not from .env: it says which file to read.
ROUTES_FILE_VAR = "QP_ROUTES_FILE"
# The setting the file feeds, its top key, and a key the workflow file must not hold.
ROUTES_SETTING = "routes"


def routes_file() -> Path | None:
    """Find the routes file: the one a variable names, else the one here.

    Returns:
        ``QP_ROUTES_FILE`` when it is set, else ``routes.yaml`` in the working
        directory when there is one, else None.

    Example:
        >>> routes_file().name  # with QP_ROUTES_FILE unset, in a use case
        'routes.yaml'
    """
    named = os.environ.get(ROUTES_FILE_VAR)
    if named:
        return Path(named)
    here = Path(ROUTES_FILE)
    return here if here.is_file() else None


def routes_file_name(path: Path | None) -> str:
    """Return the routes file as a refusal names it, and where it was looked for.

    Args:
        path: The routes file found; None for none.

    Returns:
        Its name, with the two places a command looks.

    Example:
        >>> routes_file_name(None)
        'routes file (QP_ROUTES_FILE, else ./routes.yaml; none found)'
    """
    found = repr(str(path)) if path is not None else "none found"
    return f"routes file ({ROUTES_FILE_VAR}, else ./{ROUTES_FILE}; {found})"


def routes_file_source(
    settings_cls: type[BaseSettings], path: Path | None
) -> InitSettingsSource:
    """Build the settings source that gives the routes file as ``routes``.

    Args:
        settings_cls: The settings class the source feeds.
        path: The routes file; None for none, which gives no routes.

    Returns:
        A source of one key, ``routes``, the file's list of routes.

    Raises:
        ValueError: ``path`` is not a file, cannot be read, is no YAML, or
            holds no list under ``routes``.

    Example:
        >>> routes_file_source(WorkflowSettings, Path("routes.yaml"))()
        {'routes': [{'name': 'self_contained', ...}, ...]}
    """
    routes = [] if path is None else read_routes_file(path)
    return InitSettingsSource(settings_cls, init_kwargs={ROUTES_SETTING: routes})


def read_routes_file(path: Path) -> list[object]:
    """Return the routes a file lists, as YAML read them.

    Args:
        path: The routes file.

    Returns:
        Each route's entry, checked by the settings after.

    Raises:
        ValueError: ``path`` is not a file, cannot be read, is no YAML, or is
            not a mapping whose ``routes`` is a list; the message names the
            file and the shape expected.

    Example:
        >>> [route["name"] for route in read_routes_file(Path("routes.yaml"))]
        ['self_contained', 'leans_on_history']
    """
    where = routes_file_name(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read the {where}: {exc}") from exc
    routes = data.get(ROUTES_SETTING) if isinstance(data, Mapping) else None
    if not isinstance(routes, list):
        raise ValueError(
            f"the {where} holds {data!r}; expected a mapping whose {ROUTES_SETTING!r} "
            "is a list of routes, each with a name, a description and its utterances"
        )
    return routes


def refuse_routes_key(keys: Mapping[str, object], path: Path) -> None:
    """Refuse a workflow file that holds ``routes``: they have a file of their own.

    Args:
        keys: The top-level keys of the workflow file, as YAML read them.
        path: The workflow file, named in the refusal.

    Raises:
        ValueError: The file has a ``routes`` key.

    Example:
        >>> refuse_routes_key({"graph": {}}, Path("workflow.yaml"))  # passes
    """
    if ROUTES_SETTING in keys:
        raise ValueError(
            f"workflow file {str(path)!r} has a {ROUTES_SETTING!r} key: the routes "
            f"are written in {ROUTES_FILE} beside it (or the file {ROUTES_FILE_VAR} "
            "names), not in the workflow file"
        )
