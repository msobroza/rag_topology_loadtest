"""What qp-deploy refuses in the deploy settings before anything is published.

A secret in a ``settings`` block, which would land in a launch script, and a
value the deploy uses that still holds a placeholder of the example file.

Example:
    >>> refuse_secrets(DeploySettings())
    DominoDeployError: OPENAI_API_KEY is set in deploy.yaml under defaults.settings ...
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..domain.roles import AppPart
from .apps_client import DominoDeployError
from .deploy_settings import DeploySettings, deploy_file, override_values


def refuse_secrets(settings: DeploySettings) -> None:
    """Refuse a ``settings`` block that holds a secret: it would land in a script.

    Args:
        settings: The deploy settings whose ``defaults`` and ``apps`` are checked.

    Raises:
        DominoDeployError: If a setting's name ends with ``_API_KEY``
            (``OPENAI_API_KEY``, ``COHERE_API_KEY``, ...), whatever its value.

    Example:
        >>> refuse_secrets(DeploySettings())
        DominoDeployError: OPENAI_API_KEY is set in deploy.yaml under ...
    """
    blocks = {f"apps.{part}": block for part, block in settings.apps.items()}
    for block_name, block in {"defaults": settings.defaults, **blocks}.items():
        for secret in _secrets_in(block.settings):
            raise DominoDeployError(
                f"{secret} is set in {deploy_file()} under {block_name}.settings, "
                "whose values are written into the Apps' launch scripts: set it as "
                "a Domino project or user environment variable instead"
            )


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


def _secrets_in(values: dict[str, str]) -> list[str]:
    # A name that says it holds an API key, whatever its value: "" and "domino"
    # meant something before key sources (ADR 0012), and any other is a key.
    return [name for name in values if name.endswith("_API_KEY")]
