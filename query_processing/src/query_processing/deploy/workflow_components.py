"""The workflow file and its settings, as the deploy tool reads them (ADR 0003).

A topology alias names components, so the deploy tool reads which ones the
workflow file declares: each one's name, type and backend. The overrides of
the deploy settings are values of that file, in its shape, so it checks them
with the workflow settings' own models: each key a setting of the block it is
in, each component or stage named one the file declares, then the file with
the overrides merged as the workflow app merges a variable over it. Then it
asks of those settings what an App's start-up refuses before it opens
anything, with start-up's own two checks: a name nobody registered, and a
graph that cannot run or that the chosen orchestrator does not run; and names
the variables their key sources read keys from. It reads
the file the workflow App will read, and the prompts and routes files a
command finds here, as the App finds them beside it, and nothing else: the
variables of this Workspace are not the App's. The one module of ``deploy``
that may import the workflow settings, and of the composition the two start-up
checks and ``key_source_variables`` (README rule 1).

Example:
    >>> read_workflow_components(Path("workflow.yaml"))["reranker"]
    WorkflowComponent(type='reranker', backend='local')
    >>> workflow_setting_path("QP_HTTP__RETRIES")
    ('http', 'retries')
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from ..composition.composition_root import check_workflow_graph
from ..composition.registered_names import check_registered_names, key_source_variables
from ..config.components.component_settings import (
    COMPONENT_TYPE_SETTINGS,
    Component,
    typed_by_name,
)
from ..config.components.key_blocks import KeyBlocksChecked, refuse_key_mistakes
from ..config.extensions import load_extensions
from ..config.files.prompts_file import prompts_file, prompts_file_source
from ..config.files.routes_file import routes_file, routes_file_source
from ..config.files.workflow_file import (
    WORKFLOW_FILE_VARIABLE,
    WorkflowFileNotFoundError,
    workflow_file,
    workflow_file_source,
)
from ..config.workflow_settings import WorkflowSettings
from .apps_client import DominoDeployError
from .deploy_refusals import blocks_own_refusal, first_error_in_one_line
from .deploy_settings import deploy_file

Place = tuple[str, ...]  # a value's path under overrides, its levels in order


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
        The file its ``QP_WORKFLOW_FILE`` names, else the one ``workflow_file``
        finds here, in the use case's directory, where the App runs too.

    Raises:
        DominoDeployError: If neither names a workflow file; the message says
            where it looked.

    Example:
        >>> workflow_file_for({"QP_WORKFLOW_FILE": "lexical.yaml"})
        PosixPath('lexical.yaml')
    """
    named = app_settings.get(WORKFLOW_FILE_VARIABLE)
    if named:
        return Path(named)
    try:
        return workflow_file()
    except WorkflowFileNotFoundError as exc:
        raise DominoDeployError(
            f"qp-deploy reads the workflow file the workflow App reads: {exc}"
        ) from exc


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
        >>> list(read_workflow_components(Path("workflow.yaml")))
        ['embedder', 'reranker', 'llm']
    """
    entries = typed_by_name(_workflow_data(path).get("components") or {})
    if not isinstance(entries, Mapping):
        raise DominoDeployError(_not_a_mapping(path, "its components block", entries))
    return {str(name): _component(path, name, entry) for name, entry in entries.items()}


def _workflow_data(path: Path) -> dict[str, Any]:
    # The extensions first, as the workflow app loads them: a component type
    # of the use case's is registered before its components are read.
    try:
        data = workflow_file_source(WorkflowSettings, path).yaml_data
        load_extensions(data.get("extensions"))
        return data
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise DominoDeployError(
            f"qp-deploy reads the components of the workflow file: {exc}"
        ) from exc


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
        'QP_COMPONENTS__RERANKER__SERVED__URL'
    """
    return workflow_variable(("components", name, *levels))


def workflow_variable(path: Sequence[str]) -> str:
    """Name the variable the workflow settings read a value of the workflow file from.

    Args:
        path: The value's path in the workflow file, its first level a
            top-level key.

    Returns:
        The variable: the prefix, then the path's levels joined by ``__``.

    Raises:
        ValueError: If the first level is no setting of the workflow settings.

    Example:
        >>> workflow_variable(("components", "llm", "http", "timeout_s"))
        'QP_COMPONENTS__LLM__HTTP__TIMEOUT_S'
    """
    return WorkflowSettings.variable_of(path[0], *path[1:])


def workflow_setting_path(variable: str) -> Place | None:
    """Return the path in the workflow file of the value a variable sets, if any.

    Args:
        variable: An environment variable's name, such as one of a free list.

    Returns:
        The levels of the value's path, in lower case as the workflow settings
        read them; None when the variable sets no value of the workflow file
        (a variable the launch script reads, or one of another tool).

    Example:
        >>> workflow_setting_path("QP_COMPONENTS__RERANKER__SOURCE_ID")
        ('components', 'reranker', 'source_id')
        >>> workflow_setting_path("QP_OVMS_MODELS_DIR") is None
        True
    """
    config = WorkflowSettings.model_config
    prefix = str(config.get("env_prefix", ""))
    if not variable.upper().startswith(prefix.upper()):
        return None
    delimiter = str(config.get("env_nested_delimiter", ""))
    path = tuple(variable[len(prefix) :].lower().split(delimiter))
    return path if path[0] in _fields_of(WorkflowSettings) else None


def refuse_unknown_override_keys(overrides: Mapping[str, object], path: Path) -> None:
    """Refuse a key under ``overrides`` that is no value of the workflow file.

    The keys are walked with the workflow settings' models: each one must be a
    setting of the block it is in, and under ``components`` or
    ``graph.stages`` a name the file at ``path`` declares. The settings would
    ignore an unknown variable, so a misspelt key would otherwise become a
    silent default. A stage's own settings are its stage type's, which checks
    them when the workflow app starts.

    Args:
        overrides: The ``overrides`` block of the deploy settings.
        path: The workflow file the workflow App reads.

    Raises:
        DominoDeployError: The first misspelt key, undeclared name, block that
            is not a mapping, or value left empty, with its place; first, an
            old ``api_key``, with the line start-up prints of it.

    Example:
        >>> refuse_unknown_override_keys({"htpp": {"retries": 0}}, path)
        DominoDeployError: overrides in deploy.yaml has no key 'htpp': ...
    """
    try:  # its own line, which names the key block, not "no key 'api_key'"
        refuse_key_mistakes(overrides.get("components"), WorkflowSettings.variable_of)
    except ValueError as exc:
        raise DominoDeployError(f"overrides in {deploy_file()}: {exc}") from exc
    _refuse_unknown_keys(overrides, WorkflowSettings, _workflow_data(path), (), path)


def _refuse_unknown_keys(
    value: object, model: type[BaseModel], declared: object, place: Place, file: Path
) -> None:
    fields = _fields_of(model)
    for key, below in _as_mapping(value, place).items():
        name = str(key)
        if name in fields:
            annotation, inside = fields[name].annotation, _get(declared, name)
            _refuse_unknown_below(below, annotation, inside, (*place, name), file)
        elif model.model_config.get("extra") != "allow":  # a stage keeps the rest
            raise DominoDeployError(_no_such_key_of(model, place, name, below))


def _no_such_key_of(
    model: type[BaseModel], place: Place, name: str, value: object
) -> str:
    # A renamed setting's refusal, as start-up words it, names (quoted) the key
    # it became: it says more than the list of the block's keys.
    fields = sorted(_fields_of(model))
    said = blocks_own_refusal(model, name, value)
    if any(f"'{field}'" in said for field in fields):
        return f"{_overrides_at((*place, name))} in {deploy_file()}: {said}"
    return _no_such_key(place, name, fields)


def _refuse_unknown_below(
    value: object, annotation: object, declared: object, place: Place, file: Path
) -> None:
    # An environment variable cannot say "no value", so a null has no variable.
    if value is None:
        raise DominoDeployError(
            f"{_overrides_at(place)} in {deploy_file()} has no value: give it one, "
            "or leave it out to keep the workflow file's"
        )
    if get_origin(annotation) is dict:  # entries keyed by free names
        _refuse_undeclared_names(value, get_args(annotation)[1], declared, place, file)
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _refuse_unknown_keys(value, annotation, declared, place, file)


def _refuse_undeclared_names(
    value: object, entry: object, declared: object, place: Place, file: Path
) -> None:
    names = declared if isinstance(declared, Mapping) else {}
    for key, below in _as_mapping(value, place).items():
        name = str(key)
        if name not in names:
            raise DominoDeployError(_undeclared(place, name, list(names), file))
        model = _entry_model(entry, name, names[name])
        if model is not None:  # an unknown type is the settings' to refuse
            _refuse_unknown_keys(below, model, names[name], (*place, name), file)


def _entry_model(
    annotation: object, name: str, declared: object
) -> type[BaseModel] | None:
    if annotation != Component:
        models = _models_in(annotation)
        return models[0] if len(models) == 1 else None
    # A component's settings class is found by its type, which defaults to its
    # name as in the workflow settings.
    entry = typed_by_name({name: declared})
    try:
        return COMPONENT_TYPE_SETTINGS.get(str(_get(_get(entry, name), "type")))
    except ValueError:  # an unknown type is the settings' to refuse
        return None


def _models_in(annotation: object) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for arg in get_args(annotation) for model in _models_in(arg)]


def _fields_of(model: type[BaseModel]) -> dict[str, FieldInfo]:
    # Domino's own run variables are read under Domino's names, not by a path:
    # they are no value of the workflow file, and nothing of ours sets them.
    return {
        name: field
        for name, field in model.model_fields.items()
        if field.validation_alias is None
    }


def _get(mapping: object, key: str) -> object:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def _as_mapping(value: object, place: Place) -> Mapping[object, object]:
    if isinstance(value, Mapping):
        return value
    raise DominoDeployError(
        f"{_overrides_at(place)} in {deploy_file()} is a {type(value).__name__}; "
        "expected a mapping of its keys, as in the workflow file"
    )


def _no_such_key(place: Place, name: str, known: list[str]) -> str:
    return (
        f"{_overrides_at(place)} in {deploy_file()} has no key {name!r}: an "
        "override is a value of the workflow file, in its shape (expected one "
        f"of: {_names(known)})"
    )


def _undeclared(place: Place, name: str, declared: list[str], file: Path) -> str:
    return (
        f"{_overrides_at(place)} in {deploy_file()} names {name!r}, which the "
        f"workflow file {str(file)!r} does not declare: an override changes a "
        f"value the file declares (expected one of: {_names(declared)})"
    )


def _overrides_at(place: Place) -> str:
    return ".".join(("overrides", *place))


def _names(names: list[str]) -> str:
    return ", ".join(map(repr, names)) or "none"


def refuse_override_values(overrides: Mapping[str, object], path: Path) -> None:
    """Refuse overrides that make settings the workflow app would refuse.

    The workflow file at ``path`` is read with the overrides over it, merged
    key by key as a variable is merged over it, and checked by the workflow
    settings themselves: each value's type and range, and the checks of the
    whole (a URL for a component called over HTTP, a valid run label, a
    router's routes in the routes file found here, the prompts file's
    prompts).

    Args:
        overrides: The ``overrides`` block, whose keys are already checked.
        path: The workflow file the workflow App reads.

    Raises:
        DominoDeployError: The first value the workflow settings refuse, with
            its place and what they expect.

    Example:
        >>> refuse_override_values({"http": {"retries": -1}}, path)
        DominoDeployError: ... http.retries: Input should be greater than or ...
    """
    _settings_with_overrides(overrides, path)


def refuse_what_start_up_refuses(overrides: Mapping[str, object], path: Path) -> None:
    """Refuse the settings an App would refuse when it starts, before any publish.

    The workflow file at ``path`` with the overrides over it is asked what
    ``qp-serve`` asks before it opens anything, in its order: a backend, an
    API format, a structured output method or an orchestrator nobody
    registered (``check_registered_names``), then a graph that cannot run or
    that the chosen orchestrator does not run (``check_workflow_graph``). The
    deploy tool's own exports (a topology alias's backends and URLs) are not
    part of it: ``registered_backends`` holds the backends to the registered
    ones, and a URL changes neither check.

    Args:
        overrides: The ``overrides`` block; empty when ``deploy.yaml`` has
            none, and the workflow file alone is then asked.
        path: The workflow file the workflow App reads.

    Raises:
        DominoDeployError: The first refusal, its text start-up's own, after
            the names of the two files; or, as ``refuse_override_values``, a
            value the workflow settings refuse.

    Example:
        >>> refuse_what_start_up_refuses({"orchestrator": "pydantic"}, path)
        DominoDeployError: ... QP_ORCHESTRATOR: unknown orchestrator 'pydantic' ...
    """
    settings = _settings_with_overrides(overrides, path)
    try:
        check_registered_names(settings)
        check_workflow_graph(settings)
    except ValueError as exc:  # a WorkflowGraphError is one
        raise DominoDeployError(
            f"the workflow file {str(path)!r} with the overrides of {deploy_file()} "
            f"makes settings an App refuses at start-up: {exc}"
        ) from exc


def key_variables_of(overrides: Mapping[str, object], path: Path) -> dict[str, str]:
    """Name the variables the key sources of an App's settings read keys from.

    Args:
        overrides: The ``overrides`` block, already checked with the file.
        path: The workflow file the workflow App reads.

    Returns:
        Each variable an ``env`` source names, in the file or an override, and
        the place of its ``key`` block (``key_source_variables``).

    Example:
        >>> # workflow.yaml's llm: api.key {source: env, variable: EXAMPLE_LLM_KEY}.
        >>> key_variables_of({}, Path("workflow.yaml"))
        {'EXAMPLE_LLM_KEY': 'components.llm.api.key'}
    """
    return key_source_variables(_settings_with_overrides(overrides, path))


def _settings_with_overrides(
    overrides: Mapping[str, object], path: Path
) -> WorkflowSettings:
    # Given as arguments, the overrides are the highest source, as a variable.
    arguments: dict[str, Any] = dict(overrides)
    try:
        return _file_with_overrides(path)(**arguments)
    except ValueError as exc:  # pydantic's ValidationError is one
        raise DominoDeployError(
            f"the workflow file {str(path)!r} with the overrides of {deploy_file()} "
            f"makes settings the workflow app refuses: {first_error_in_one_line(exc)}"
        ) from exc


def _file_with_overrides(path: Path) -> type[WorkflowSettings]:
    # The workflow settings with the overrides, given as arguments, over the
    # file at path, and nothing else: this Workspace's variables and .env are
    # not the App's. Pydantic passes every source by keyword, so the names stay.
    # The prompts and routes files are read as well, as a command finds them
    # here, where the App runs: the settings refuse a router's routes against
    # its catalogue, so without the routes file every workflow file that
    # declares a router was refused; and start-up's graph check refuses a
    # stage that names a prompt the prompts file does not hold.
    class FileWithOverrides(WorkflowSettings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            del env_settings, dotenv_settings, file_secret_settings
            # The key blocks checked as start-up checks them, never quoting one.
            return (
                init_settings,
                workflow_file_source(settings_cls, path),
                prompts_file_source(settings_cls, prompts_file()),
                routes_file_source(settings_cls, routes_file()),
                KeyBlocksChecked(settings_cls, WorkflowSettings.variable_of),
            )

    return FileWithOverrides
