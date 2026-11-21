"""What qp-deploy refuses in the deploy settings before anything is published.

A secret in a ``settings`` block, which would land in a launch script, a
value the deploy uses that still holds a placeholder of the example file, and
no directory of wheels for the Apps' install. And the words of what the
workflow settings refuse of an override, which ``workflow_components.py``
quotes: they live here, out of that module, for the 500-line limit of a file.

Example:
    >>> refuse_secrets(DeploySettings(defaults={"settings": {"X_API_KEY": ""}}))
    DominoDeployError: X_API_KEY is set in deploy.yaml under defaults.settings ...
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..domain.roles import AppPart
from .apps_client import DominoDeployError
from .deploy_settings import DeploySettings, deploy_file, override_values

# The wheels of doclang's core, as make doclang-wheel names one and the launch
# script looks for it (domino/app.sh).
DOCLANG_WHEELS = "doclang_domain_core-*.whl"


def refuse_secrets(settings: DeploySettings) -> None:
    """Refuse a ``settings`` block that holds a secret: it would land in a script.

    Args:
        settings: The deploy settings whose ``defaults`` and ``apps`` are checked.

    Raises:
        DominoDeployError: If a setting's name ends with ``_API_KEY``
            (``X_API_KEY`` and the like), whatever its value.

    Example:
        >>> refuse_secrets(DeploySettings(defaults={"settings": {"X_API_KEY": ""}}))
        DominoDeployError: X_API_KEY is set in deploy.yaml under ...
    """
    blocks = {f"apps.{part}": block for part, block in settings.apps.items()}
    for block_name, block in {"defaults": settings.defaults, **blocks}.items():
        for secret in _secrets_in(block.settings):
            raise DominoDeployError(
                f"{secret} is set in {deploy_file()} under {block_name}.settings, "
                "whose values are written into the Apps' launch scripts: set it as "
                "a Domino project or user environment variable instead"
            )


def refuse_no_wheels(settings: DeploySettings, *, on_this_machine: bool = True) -> None:
    """Refuse settings that name no directory of wheels the Apps can install from.

    Every App installs at its start, and doclang's core, which the core depends
    on, is on no index: without one wheel of it in an absolute directory that
    exists here, as the Workspace mounts the Apps' Dataset, each App would stop
    at its start, after the deploy had published it. As ``ovms_bundle`` is held.

    Args:
        settings: The deploy settings.
        on_this_machine: Whether the directory is looked at here. A dry run
            passes False: it runs where no Dataset is mounted, on a laptop, and
            holds the setting to what it can tell there, a path that is set
            and absolute.

    Raises:
        DominoDeployError: If ``bootstrap_wheels`` is blank or relative, or,
            looked at here, names no directory or one without exactly one
            wheel of doclang's core.

    Example:
        >>> refuse_no_wheels(DeploySettings(bootstrap_wheels=""))
        DominoDeployError: bootstrap_wheels is not set in deploy.yaml ...
    """
    directory = settings.bootstrap_wheels.strip()
    if not directory:
        raise DominoDeployError(
            f"bootstrap_wheels is not set in {deploy_file()}: set it (or "
            "QP_DEPLOY_BOOTSTRAP_WHEELS) to the directory on the Dataset holding "
            "the wheel of doclang's core, which make doclang-wheel builds; every "
            "App installs doclang's core from there, as no package index has it"
        )
    unusable = _why_the_apps_cannot_install_from(Path(directory), on_this_machine)
    if unusable:
        raise DominoDeployError(
            f"bootstrap_wheels={directory} {unusable}; set it to the absolute "
            "directory on the Dataset, as the Apps read it, holding the one wheel "
            "of doclang's core make doclang-wheel built"
        )


def _why_the_apps_cannot_install_from(directory: Path, on_this_machine: bool) -> str:
    # "" when the directory holds what an App's install needs, as far as this
    # machine can tell: only a deploy looks at the directory itself.
    if not directory.is_absolute():
        return "is a relative path, which each App would read from its own directory"
    if not on_this_machine:
        return ""
    if not directory.is_dir():
        return "does not exist"
    wheels = sorted(path.name for path in directory.glob(DOCLANG_WHEELS))
    if not wheels:
        return f"holds no wheel of doclang's core ({DOCLANG_WHEELS})"
    if len(wheels) > 1:
        return f"holds {len(wheels)} wheels of doclang's core ({', '.join(wheels)})"
    return ""


# "<dataset>", "<environment id>": the example file's placeholders. No spaces at the
# edges, so a comparison such as "a < b" is not taken for one.
_PLACEHOLDER = re.compile(r"<[^<>\s][^<>]*[^<>\s]>|<[^<>\s]>")


def refuse_placeholders(settings: DeploySettings, parts: list[AppPart]) -> None:
    """Refuse a value the deploy of ``parts`` uses that still holds a placeholder.

    Args:
        settings: The deploy settings.
        parts: The Apps being deployed; other Apps' blocks are not checked.

    Raises:
        DominoDeployError: If a value holds a placeholder such as ``<dataset>``.

    Example:
        >>> refuse_placeholders(DeploySettings(), ["workflow"])
        DominoDeployError: ... '<dataset>' ...
    """
    for key, block, value in _values_used(settings, parts):
        if _PLACEHOLDER.search(value):
            raise DominoDeployError(
                f"{key} in {block} of {deploy_file()} still holds the example "
                f"placeholder {value!r}: replace it with a real value"
            )


def _values_used(
    settings: DeploySettings, parts: list[AppPart]
) -> Iterator[tuple[str, str, str]]:
    yield "ovms_bundle", "the top level", settings.ovms_bundle
    blocks = [("defaults", settings.defaults)] + [
        (f"apps.{part}", settings.apps[part]) for part in parts if part in settings.apps
    ]
    for name, block in blocks:
        yield "environment_id", name, block.environment_id or ""
        yield "environment_revision_id", name, block.environment_revision_id or ""
        yield "hardware_tier_id", name, block.hardware_tier_id or ""
        yield "hardware_tier", name, block.hardware_tier or ""
        for key, value in block.settings.items():
            yield key, f"{name}.settings", value
    # Every App gets the overrides, so any deploy uses each of them.
    for override in override_values(settings.overrides):
        yield override.place, "overrides", override.text
    # Last, so that the example copied as it is still stops on its environment.
    yield "bootstrap_wheels", "the top level", settings.bootstrap_wheels


def _secrets_in(values: dict[str, str]) -> list[str]:
    # A name that says it holds an API key, whatever its value: "" and "domino"
    # meant something before key sources (ADR 0012), and any other is a key.
    return [name for name in values if name.endswith("_API_KEY")]


def first_error_in_one_line(exc: ValueError) -> str:
    """Word the first thing a settings model refused, with its place and value.

    Args:
        exc: What reading the settings raised; pydantic's ``ValidationError``
            is a ``ValueError``.

    Returns:
        The first error's place, its message and the value given, a mapping
        or a list left out; the message alone for a check of the whole, which
        names its variables itself; ``str(exc)`` for any other ``ValueError``.

    Example:
        >>> first_error_in_one_line(refused)  # WorkflowSettings(http={"retries": -1})
        'http.retries: Input should be greater than or equal to 0 (got -1)'
    """
    if not isinstance(exc, ValidationError):
        return str(exc)
    error = exc.errors()[0]
    loc = list(error["loc"])
    if not loc:  # a check of the whole names its variables itself
        return str(error["msg"])
    got = error["input"]
    shown = "" if isinstance(got, (Mapping, list)) else f" (got {got!r})"
    return f"{'.'.join(map(str, loc))}: {error['msg']}{shown}"


def blocks_own_refusal(model: type[BaseModel], name: str, value: object) -> str:
    """Return a settings block's own words for a key it refuses.

    Args:
        model: The settings block's model.
        name: The key given.
        value: Its value.

    Returns:
        What a validator of the block says of the key, such as the new name
        of a renamed setting; "" if the block takes the key, or refuses it
        with no words of its own.

    Example:
        >>> blocks_own_refusal(RerankerLocalSettings, "batch_size", 8)
        ''
    """
    try:
        model.model_validate({name: value})
    except ValidationError as error:
        said = [e["msg"] for e in error.errors() if e["type"] == "value_error"]
        return said[0].removeprefix("Value error, ") if said else ""
    return ""
