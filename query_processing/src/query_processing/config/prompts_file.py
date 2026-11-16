"""The prompts file: where it is found, and the settings source that reads it.

A use case writes its prompts in ``prompts.yaml``, beside its workflow file
(spec 4): each top-level key is a prompt's name, its value a list of messages
and placeholders (``domain.prompts``). It feeds the top-level setting
``prompts`` of ``WorkflowSettings``, below the environment, so a variable named
by a prompt's path, such as ``QP_PROMPTS__ANSWER_FROM_CONTEXT``, replaces the
whole prompt, written as JSON. A command finds the file as it finds the
workflow file: the one ``QP_PROMPTS_FILE`` names, else ``prompts.yaml`` in the
working directory. A use case with neither has no prompts, and a stage that
names one is refused at start-up. The workflow file holds no ``prompts`` key.

Example:
    >>> source = prompts_file_source(WorkflowSettings, prompts_file())
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, InitSettingsSource

PROMPTS_FILE = "prompts.yaml"  # in the working directory
# Read from the environment only, not from .env: it says which file to read.
PROMPTS_FILE_VAR = "QP_PROMPTS_FILE"
# The setting the file feeds, a key the workflow file must not hold.
PROMPTS_SETTING = "prompts"


def prompts_file() -> Path | None:
    """Find the prompts file: the one a variable names, else the one here.

    Returns:
        ``QP_PROMPTS_FILE`` when it is set, else ``prompts.yaml`` in the
        working directory when there is one, else None.

    Example:
        >>> prompts_file().name  # with QP_PROMPTS_FILE unset, in a use case
        'prompts.yaml'
    """
    named = os.environ.get(PROMPTS_FILE_VAR)
    if named:
        return Path(named)
    here = Path(PROMPTS_FILE)
    return here if here.is_file() else None


def prompts_file_source(
    settings_cls: type[BaseSettings], path: Path | None
) -> InitSettingsSource:
    """Build the settings source that gives the prompts file as ``prompts``.

    Args:
        settings_cls: The settings class the source feeds.
        path: The prompts file; None for none, which gives no prompts.

    Returns:
        A source of one key, ``prompts``, the file's mapping of prompts by name.

    Raises:
        ValueError: ``path`` is not a file, cannot be read, is no YAML, or does
            not map each prompt's name to its list of messages.

    Example:
        >>> prompts_file_source(WorkflowSettings, Path("prompts.yaml"))()
        {'prompts': {'answer_from_context': [...]}}
    """
    prompts = {} if path is None else read_prompts_file(path)
    return InitSettingsSource(settings_cls, init_kwargs={PROMPTS_SETTING: prompts})


def read_prompts_file(path: Path) -> dict[str, object]:
    """Return the prompts a file holds, by name, as YAML read them.

    Args:
        path: The prompts file.

    Returns:
        Each prompt's name with its value, checked by the settings after.

    Raises:
        ValueError: ``path`` is not a file, cannot be read, is no YAML, or is
            no mapping; the message names the file and the shape expected.

    Example:
        >>> list(read_prompts_file(Path("prompts.yaml")))
        ['answer_from_context']
    """
    where = f"prompts file {str(path)!r} ({PROMPTS_FILE_VAR}, else ./{PROMPTS_FILE})"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read the {where}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(
            f"the {where} holds a {type(data).__name__}; expected a mapping of "
            "each prompt's name to its list of messages"
        )
    return {str(name): prompt for name, prompt in data.items()}


def refuse_prompts_key(keys: Mapping[str, object], path: Path) -> None:
    """Refuse a workflow file that holds ``prompts``: they have a file of their own.

    Args:
        keys: The top-level keys of the workflow file, as YAML read them.
        path: The workflow file, named in the refusal.

    Raises:
        ValueError: The file has a ``prompts`` key.

    Example:
        >>> refuse_prompts_key({"graph": {}}, Path("workflow.yaml"))  # passes
    """
    if PROMPTS_SETTING in keys:
        raise ValueError(
            f"workflow file {str(path)!r} has a {PROMPTS_SETTING!r} key: the "
            f"prompts are written in {PROMPTS_FILE} beside it (or the file "
            f"{PROMPTS_FILE_VAR} names), not in the workflow file"
        )
