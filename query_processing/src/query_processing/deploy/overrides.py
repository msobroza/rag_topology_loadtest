"""The overrides: this site's values of the workflow file, handed to every App.

``deploy.yaml`` writes them under ``overrides`` in the workflow file's own
shape, so that no one hand-writes a nested variable name. Before anything is
published they are checked against the workflow settings (each key a value of
the file, no secret, each value one the workflow app accepts), then the
workflow file with them is asked what an App's start-up refuses (a name
nobody registered, a key block its source does not read, a graph its
orchestrator does not run), even with no overrides at all. A ``key`` block may
be overridden, since it holds no key; an old ``api_key`` gets start-up's line.
A workflow setting written in a free list (``defaults.settings``,
``apps.<part>.settings``) is refused: those lists are for the variables the
launch scripts read; so is a variable an ``env`` key source of the settings
with the overrides names, which holds a key. Each
App's launch script then exports every override under the variable the
workflow settings read for its path, so nothing new is read on the App's side.
The deploy tool's own exports win over an override, and an override wins over
the workflow file.

Example:
    >>> override_exports(DeploySettings(overrides={"http": {"retries": 0}}))
    {'QP_HTTP__RETRIES': '0'}
"""

from __future__ import annotations

from pathlib import Path

from ..domain.roles import WORKFLOW_APP
from .apps_client import DominoDeployError
from .deploy_settings import DeploySettings, deploy_file, override_values
from .workflow_components import (
    key_variables_of,
    refuse_override_values,
    refuse_unknown_override_keys,
    refuse_what_start_up_refuses,
    workflow_file_for,
    workflow_setting_path,
    workflow_variable,
)


def check_overrides(settings: DeploySettings) -> None:
    """Refuse overrides the workflow App could not start with, before publishing.

    Then, even when ``deploy.yaml`` has no overrides, the workflow file with
    them is asked what an App's start-up refuses; and a free list may not give
    a value to a variable an ``env`` key source of those settings names.

    Args:
        settings: The deploy settings; the workflow App's free list names the
            workflow file the overrides are values of.

    Raises:
        DominoDeployError: If a key is no value of the workflow file (naming
            it and its place), a value is a secret, the workflow settings
            refuse a value, or an App's start-up would refuse the settings
            (start-up's own line, after the names of the two files), or a free
            list sets a variable a key source reads a key from.

    Example:
        >>> check_overrides(DeploySettings(overrides={"htpp": {"retries": 0}}))
        DominoDeployError: overrides in deploy.yaml has no key 'htpp': ...
    """
    path = workflow_file_for(settings.for_app(WORKFLOW_APP).settings)
    if settings.overrides:
        refuse_unknown_override_keys(settings.overrides, path)
        _refuse_secrets_in_overrides(settings)
        refuse_override_values(settings.overrides, path)
    # Even with no overrides: the workflow file alone can name an orchestrator
    # nobody registered, or hold a graph its orchestrator does not run.
    refuse_what_start_up_refuses(settings.overrides, path)
    _refuse_key_variables_in_free_lists(settings, path)


def _refuse_secrets_in_overrides(settings: DeploySettings) -> None:
    # The rule of the free lists, whatever the value: a stage keeps keys of its
    # own, which no model of the workflow settings refuses as unknown.
    for value in override_values(settings.overrides):
        variable = workflow_variable(value.path)
        if not variable.endswith("_API_KEY"):
            continue
        raise DominoDeployError(
            f"overrides.{value.place} is set in {deploy_file()}, whose values are "
            f"written into the Apps' launch scripts: set {variable} as a Domino "
            "project or user environment variable instead"
        )


def _refuse_key_variables_in_free_lists(settings: DeploySettings, path: Path) -> None:
    # As the Apps will read the key blocks: the file's with the overrides over it.
    named = key_variables_of(settings.overrides, path)
    for block_name, variable, _value in _free_list_variables(settings):
        if variable not in named:
            continue
        raise DominoDeployError(
            f"{variable} is set in {deploy_file()} under {block_name}.settings, "
            "whose values are written into the Apps' launch scripts, but the key "
            f"source of {named[variable]} reads an API key from it: set it as a "
            "Domino project or user environment variable instead"
        )


def refuse_workflow_settings_in_free_lists(settings: DeploySettings) -> None:
    """Refuse a value of the workflow file written as a variable in a free list.

    Args:
        settings: The deploy settings whose ``defaults`` and ``apps`` blocks
            are checked, every one, deployed or not.

    Raises:
        DominoDeployError: The first such variable, with the path under
            ``overrides`` where it belongs.

    Example:
        >>> refuse_workflow_settings_in_free_lists(DeploySettings(defaults={
        ...     "settings": {"QP_HTTP__RETRIES": "0"}}))
        DominoDeployError: QP_HTTP__RETRIES is set ... write it under overrides ...
    """
    for block_name, variable, value in _free_list_variables(settings):
        path = workflow_setting_path(variable)
        if path is None:
            continue
        where = (
            f"{variable} is set in {deploy_file()} under {block_name}.settings, "
            "the variables the launch scripts read, but it is a value of the "
            "workflow file"
        )
        refused = _refusal_under_overrides(settings, path, value)
        if refused:
            raise DominoDeployError(
                f"{where}, and under overrides it is refused too: {refused}"
            )
        raise DominoDeployError(
            f"{where}: write it under overrides as {'.'.join(path)}, in "
            "the workflow file's shape, and every App gets it"
        )


def _refusal_under_overrides(
    settings: DeploySettings, path: tuple[str, ...], value: str
) -> str:
    # Written under overrides, a key the workflow file's block refuses (an old
    # name) would be refused there in turn: its line is given at once. "" when
    # no workflow file is found here, as the place alone can then be said.
    try:
        file = workflow_file_for(settings.for_app(WORKFLOW_APP).settings)
    except DominoDeployError:
        return ""
    override: dict[str, object] = {path[-1]: value}
    for level in reversed(path[:-1]):
        override = {level: override}
    try:
        refuse_unknown_override_keys(override, file)
    except DominoDeployError as refusal:
        return str(refusal)
    return ""


def _free_list_variables(settings: DeploySettings) -> list[tuple[str, str, str]]:
    blocks = {f"apps.{part}": block for part, block in settings.apps.items()}
    return [
        (block_name, variable, value)
        for block_name, block in {"defaults": settings.defaults, **blocks}.items()
        for variable, value in block.settings.items()
    ]


def override_exports(settings: DeploySettings) -> dict[str, str]:
    """Name the variable of each override and the text it is exported as.

    Args:
        settings: The deploy settings, their overrides already checked.

    Returns:
        Each override's variable, the one the workflow settings read for its
        path, and its text, in the order of ``deploy.yaml``.

    Example:
        >>> override_exports(DeploySettings(overrides={
        ...     "components": {"llm": {"http": {"timeout_s": 120}}}}))
        {'QP_COMPONENTS__LLM__HTTP__TIMEOUT_S': '120'}
    """
    return {
        workflow_variable(value.path): value.text
        for value in override_values(settings.overrides)
    }
