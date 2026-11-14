"""The settings source that reads the prompts from a YAML file.

The prompts have no Python default: they come from ``prompts.yaml`` shipped
with this package, or from the file ``RAG_PROMPTS_FILE`` names. ``RagSettings``
puts this source last, so ``RAG_SYSTEM_PROMPT`` / ``RAG_USER_PROMPT`` and
constructor arguments still override it.

Example:
    >>> source = prompts_source(RagSettings, prompts_path())
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, YamlConfigSettingsSource

PACKAGED_PROMPTS = Path(__file__).with_name("prompts.yaml")


def prompts_path() -> Path:
    """Name the prompts file: ``RAG_PROMPTS_FILE`` when set, else the packaged one.

    Returns:
        The path of the prompts YAML file.

    Example:
        >>> prompts_path().name  # with RAG_PROMPTS_FILE unset
        'prompts.yaml'
    """
    override = os.environ.get("RAG_PROMPTS_FILE")
    return Path(override) if override else PACKAGED_PROMPTS


def prompts_source(
    settings_cls: type[BaseSettings], path: Path
) -> YamlConfigSettingsSource:
    """Build the YAML settings source over the prompts file.

    A missing or unreadable file is an error: pydantic-settings would silently skip
    a missing one, leaving the prompts "Field required" without saying which file
    was meant.

    Args:
        settings_cls: The settings class the source feeds.
        path: The prompts YAML file.

    Returns:
        The YAML source over ``path``.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        OSError: If ``path`` cannot be read.

    Example:
        >>> prompts_source(RagSettings, Path("missing.yaml"))
        FileNotFoundError: prompts file 'missing.yaml' does not exist
    """
    if not path.is_file():
        raise FileNotFoundError(f"prompts file {str(path)!r} does not exist")
    try:
        return YamlConfigSettingsSource(settings_cls, yaml_file=path)
    except OSError as exc:
        raise OSError(f"cannot read prompts file {str(path)!r}: {exc}") from exc
