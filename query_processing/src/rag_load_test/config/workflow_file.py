"""The workflow file: where it is found, and the settings source that reads it.

Its top-level keys are settings of ``RagSettings``: the ``components`` and
the ``graph`` blocks (ADR 0004), and the blocks of what the components share
(``local_models``, ``calibration``, ``http``, ``domino``). It is found as the
deploy settings find
theirs: the file ``RAG_WORKFLOW_FILE`` names, else ``workflow.yaml`` in the
working directory, else the file that ships with this package, the RAG's.
``RagSettings`` puts the file below the environment and ``.env``, so a variable
named by a value's path, such as ``RAG_GRAPH__STAGES__RERANK__TOP_K``, wins
over it.

Example:
    >>> source = workflow_file_source(RagSettings, workflow_file())
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, YamlConfigSettingsSource

from .component_settings import alias_of

PACKAGED_WORKFLOW = Path(__file__).with_name("workflow.yaml")
WORKFLOW_FILE = "workflow.yaml"  # in the working directory
# Read from the environment only, not from .env: it says which files to read.
WORKFLOW_FILE_VARIABLE = "RAG_WORKFLOW_FILE"


def workflow_file() -> Path:
    """Find the workflow file: the one a variable names, else here, else packaged.

    The variable comes first, so that a variant can be tried from a directory
    that holds a ``workflow.yaml`` without moving it.

    Returns:
        ``RAG_WORKFLOW_FILE`` when it is set, else ``workflow.yaml`` in the
        working directory when there is one, else the packaged file.

    Example:
        >>> workflow_file().name  # with RAG_WORKFLOW_FILE unset
        'workflow.yaml'
    """
    named = os.environ.get(WORKFLOW_FILE_VARIABLE)
    if named:
        return Path(named)
    here = Path(WORKFLOW_FILE)
    return here if here.is_file() else PACKAGED_WORKFLOW


def workflow_file_source(
    settings_cls: type[BaseSettings], path: Path
) -> YamlConfigSettingsSource:
    """Build the YAML settings source over the workflow file.

    A missing file is an error: pydantic-settings would silently skip it,
    leaving the graph "Field required" without saying which file was meant. A
    top-level key that is no setting is an error too: the settings ignore an
    unknown variable, so a typo in the file would become a silent default.

    Args:
        settings_cls: The settings class the source feeds.
        path: The workflow file.

    Returns:
        The YAML source over ``path``.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        OSError: If ``path`` cannot be read.
        ValueError: If a top-level key is not a field of ``settings_cls``.

    Example:
        >>> workflow_file_source(RagSettings, Path("missing.yaml"))
        FileNotFoundError: workflow file 'missing.yaml' does not exist
    """
    if not path.is_file():
        raise FileNotFoundError(f"workflow file {str(path)!r} does not exist")
    try:
        source = YamlConfigSettingsSource(settings_cls, yaml_file=path)
    except OSError as exc:
        raise OSError(f"cannot read workflow file {str(path)!r}: {exc}") from exc
    unknown = unknown_keys(source.yaml_data, settings_cls)
    if unknown:
        raise ValueError(
            f"workflow file {str(path)!r} has unknown key(s) "
            f"{', '.join(map(repr, unknown))}: each top-level key is a setting, "
            "such as 'components' or 'graph'"
        )
    return source


def unknown_keys(keys: Iterable[object], settings_cls: type[BaseSettings]) -> list[str]:
    """Return the keys that are not fields of ``settings_cls``, sorted.

    Args:
        keys: The top-level keys of a workflow file, as YAML read them.
        settings_cls: The settings class the file feeds.

    Returns:
        The keys no field of ``settings_cls`` is named after, as text.

    Example:
        >>> unknown_keys(["graph", "grpah"], RagSettings)
        ['grpah']
    """
    fields = settings_cls.model_fields
    return sorted(str(key) for key in keys if key not in fields)


def declared_alias(component: str, path: Path | None = None) -> str:
    """Return a component's alias as the workflow file declares it.

    It reads the file alone, so a client that builds no workflow settings (the
    load-test client) sends the alias a model server serves the model under.

    Args:
        component: The component's name, such as ``reranker``.
        path: The workflow file; None for the one ``workflow_file`` finds.

    Returns:
        Its ``alias``, else the last part of its ``source_id``; "" when the
        file declares no such component.

    Example:
        >>> declared_alias("reranker")
        'crossencoder-camembert-base-mmarcoFR'
    """
    text = (path or workflow_file()).read_text(encoding="utf-8")
    components = (yaml.safe_load(text) or {}).get("components") or {}
    entry = components.get(component)
    if not isinstance(entry, Mapping):
        return ""
    return str(entry.get("alias") or alias_of(str(entry.get("source_id") or "")))
