"""The App ids ``rag-deploy`` has seen, so an App is redeployed, never made twice.

The Apps API's ``name`` filter is not documented as exact, and a second App with
the same name would split the traffic: the id found or created once is reused.

Example:
    >>> cache = AppIdCache(".rag-deploy-cache.json")
    >>> cache.remember("proj-1", "rag-workflow", "app-7")
"""

from __future__ import annotations

import json
import os
from typing import TypeGuard

from .apps_client import DominoDeployError

APP_ID_CACHE_FILE = ".rag-deploy-cache.json"  # in the working directory


class AppIdCache:
    """``{project id: {App name: App id}}``, kept in a JSON file.

    Args:
        path: The JSON file; None keeps the ids in memory only. A file that is
            not such a table raises ``DominoDeployError``.
    """

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._ids = _read(path) if path is not None else {}

    def get(self, project_id: str, name: str) -> str | None:
        """Look up the id of an App seen before.

        Args:
            project_id: The App's project.
            name: The App's name.

        Returns:
            Its id, or None if it has not been seen.
        """
        return self._ids.get(project_id, {}).get(name)

    def remember(self, project_id: str, name: str, app_id: str) -> None:
        """Record an App's id, and write the file.

        Args:
            project_id: The App's project.
            name: The App's name.
            app_id: The id Domino gave it.
        """
        self._ids.setdefault(project_id, {})[name] = app_id
        self._write()

    def forget(self, project_id: str, name: str) -> None:
        """Drop an App's id, e.g. once it is deleted in Domino, and write the file.

        Args:
            project_id: The App's project.
            name: The App's name.
        """
        self._ids.get(project_id, {}).pop(name, None)
        self._write()

    def _write(self) -> None:
        if self._path is not None:
            with open(self._path, "w", encoding="utf-8") as file:
                json.dump(self._ids, file, indent=2, sort_keys=True)


def _read(path: str) -> dict[str, dict[str, str]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            ids = json.load(file)
    except ValueError:
        ids = None
    if not _is_id_table(ids):
        raise DominoDeployError(
            f"{path} is not an App id cache ({{project id: {{App name: App id}}}}): "
            "delete it and rag-deploy finds the Apps by name again"
        )
    return ids


def _is_id_table(value: object) -> TypeGuard[dict[str, dict[str, str]]]:
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(names, dict)
        and all(isinstance(app_id, str) for app_id in names.values())
        for names in value.values()
    )
