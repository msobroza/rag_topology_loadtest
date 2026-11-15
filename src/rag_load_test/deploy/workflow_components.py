"""The components of the workflow file, as the deploy tool reads them (ADR 0003).

A topology alias names components, so the deploy tool reads which ones the
workflow file declares: each one's name, type and backend. It reads the file
the workflow App will read, and the file alone: the variables of this
Workspace and the start-up checks of the workflow app are not the deploy
tool's. The one module of ``deploy`` that may import the workflow settings.

Example:
    >>> read_workflow_components(PACKAGED_WORKFLOW)["reranker"]
    WorkflowComponent(type='reranker', backend='local')
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config.component_settings import typed_by_name
from ..config.rag_settings import RagSettings
from ..config.workflow_file import (
    WORKFLOW_FILE_VARIABLE,
    workflow_file,
    workflow_file_source,
)
from .apps_client import DominoDeployError


@dataclass(frozen=True)
class WorkflowComponent:
    """One component of the workflow file: its type and the backend it states."""

    type: str
    backend: str


def workflow_file_for(app_settings: Mapping[str, str]) -> Path:
    """Find the workflow file the workflow App reads.

    Args:
        app_settings: The workflow App's settings, as its launch script exports
            them.

    Returns:
        The file its ``RAG_WORKFLOW_FILE`` names, else the one ``workflow_file``
        finds here, where the App starts too (the repository root).

    Example:
        >>> workflow_file_for({"RAG_WORKFLOW_FILE": "lexical.yaml"})
        PosixPath('lexical.yaml')
    """
    named = app_settings.get(WORKFLOW_FILE_VARIABLE)
    return Path(named) if named else workflow_file()


def read_workflow_components(path: Path) -> dict[str, WorkflowComponent]:
    """Read the components the workflow file at ``path`` declares, in its order.

    Args:
        path: The workflow file.

    Returns:
        Each component by name, its type defaulting to its name as in the
        workflow settings.

    Raises:
        DominoDeployError: If the file is missing, cannot be read, has an
            unknown top-level key, or its ``components`` block is not a mapping
            of names to mappings.

    Example:
        >>> list(read_workflow_components(PACKAGED_WORKFLOW))
        ['embedder', 'reranker', 'llm']
    """
    try:
        data = workflow_file_source(RagSettings, path).yaml_data
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise DominoDeployError(
            f"rag-deploy reads the components of the workflow file: {exc}"
        ) from exc
    entries = typed_by_name(data.get("components") or {})
    if not isinstance(entries, Mapping):
        raise DominoDeployError(_not_a_mapping(path, "its components block", entries))
    return {str(name): _component(path, name, entry) for name, entry in entries.items()}


def _component(path: Path, name: object, entry: object) -> WorkflowComponent:
    if not isinstance(entry, Mapping):
        raise DominoDeployError(_not_a_mapping(path, f"component {name!r}", entry))
    return WorkflowComponent(str(entry["type"]), str(entry.get("backend") or ""))


def _not_a_mapping(path: Path, what: str, value: object) -> str:
    return (
        f"workflow file {str(path)!r}: {what} is {value!r}; expected a mapping, "
        "such as {embedder: {backend: local, source_id: ...}}"
    )


def component_variable(name: str, *levels: str) -> str:
    """Name the variable that sets a value of a component of the workflow file.

    Args:
        name: The component's name.
        *levels: The path below it, such as ``served``, ``url``.

    Returns:
        The variable, its path's levels joined by ``__``, as the workflow app
        reads it.

    Example:
        >>> component_variable("reranker", "served", "url")
        'RAG_COMPONENTS__RERANKER__SERVED__URL'
    """
    return RagSettings.variable_of("components", name, *levels)
