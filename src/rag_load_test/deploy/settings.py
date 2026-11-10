"""rag-deploy's settings: ``deploy.yaml``, overridden by ``RAG_DEPLOY_*`` variables.

Top-level keys say what to deploy and how ``rag-deploy`` talks to Domino;
``defaults`` and ``apps.<part>`` say how each App runs. ``deploy.example.yaml``
documents every key.

Example::

    DeploySettings().for_app("workflow").kind   # "agent"
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from ..domain.topology import Topology
from .apps_client import DominoDeployError, WaitLimits

AppPart = Literal["embedder", "reranker", "workflow"]
AppKind = Literal["app", "agent"]
ModelServer = Literal["fastapi", "ovms"]
DEPLOY_FILE = (
    "deploy.yaml"  # in the working directory, unless RAG_DEPLOY_FILE names one
)
# The orchestrator is published as an Agent, the model servers as plain Apps.
DEFAULT_KIND: dict[AppPart, AppKind] = {
    "embedder": "app",
    "reranker": "app",
    "workflow": "agent",
}


def deploy_file() -> str:
    """The settings file: ``RAG_DEPLOY_FILE``, else ``deploy.yaml`` here.

    Example::

        deploy_file()   # "deploy.yaml"
    """
    return os.environ.get("RAG_DEPLOY_FILE") or DEPLOY_FILE


class _Block(BaseModel):
    # A misspelt key must be reported, not silently ignored.
    model_config = ConfigDict(extra="forbid")


class Autoscaling(_Block):
    """Replicas of one App instance (pods behind the same instance id)."""

    min_replicas: int = 1
    max_replicas: int
    target_cpu_pct: int | None = None

    @model_validator(mode="after")
    def _possible(self) -> Autoscaling:
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"max_replicas {self.max_replicas} is below "
                f"min_replicas {self.min_replicas}"
            )
        if self.target_cpu_pct is not None and not 1 <= self.target_cpu_pct <= 100:
            raise ValueError(f"target_cpu_pct {self.target_cpu_pct} is outside 1..100")
        return self


class AppBlock(_Block):
    """``defaults`` or one ``apps.<part>`` entry; None = not set here."""

    kind: AppKind | None = None
    environment_id: str | None = None
    hardware_tier_id: str | None = None
    autoscaling: Autoscaling | None = None
    settings: dict[str, str] = {}

    @field_validator("settings", mode="before")
    @classmethod
    def _scalars_as_text(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value  # pydantic reports the wrong shape
        return {name: _as_text(name, scalar) for name, scalar in value.items()}


def _as_text(name: str, value: object) -> str:
    if isinstance(value, bool):  # YAML true/false, exported as the shell spells them
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError(f"{name} must be a string, a number or a boolean, not {value!r}")


@dataclass(frozen=True)
class AppSettings:
    """One App's settings once its own values and the defaults are merged."""

    part: AppPart
    kind: AppKind
    environment_id: str
    hardware_tier_id: str
    autoscaling: Autoscaling | None
    settings: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"rag-{self.part}"


class DeploySettings(BaseSettings):
    """``deploy.yaml``; ``RAG_DEPLOY_<KEY>`` overrides a top-level key."""

    model_config = SettingsConfigDict(env_prefix="RAG_DEPLOY_", extra="forbid")

    topology: Topology = "split-all"
    model_server: ModelServer = "fastapi"
    project_id: str = ""  # "" = DOMINO_PROJECT_ID, which Domino sets in a Workspace
    ovms_bundle: str = ""  # required for ``ovms``: bin/ovms + lib/, or its .tar.gz
    domino_host: str = ""  # "" = the scheme and host of each App's own url
    wait_timeout_s: float = 600.0
    poll_interval_s: float = 5.0
    # The 6.2 API has no start call: creating a version is assumed to start an
    # instance. One that has still not appeared after this long is reported.
    no_instance_grace_s: float = 60.0
    http_timeout_s: float = 30.0
    http_retries: int = 2
    run_token_url: str = "http://localhost:8899/access-token"  # a run's token service
    defaults: AppBlock = AppBlock()
    apps: dict[AppPart, AppBlock] = {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = YamlConfigSettingsSource(settings_cls, yaml_file=deploy_file())
        return init_settings, env_settings, yaml_file

    @model_validator(mode="after")
    def _project_from_the_workspace(self) -> DeploySettings:
        # The example's `project_id: ""` means "not set", not "".
        self.project_id = self.project_id or os.environ.get("DOMINO_PROJECT_ID", "")
        return self

    def for_app(self, part: AppPart) -> AppSettings:
        """``rag-<part>``'s settings: its own value, else the default.

        Example::

            DeploySettings().for_app("reranker").hardware_tier_id   # "cpu-16"
        """
        own, shared = self.apps.get(part, AppBlock()), self.defaults
        return AppSettings(
            part=part,
            kind=own.kind or shared.kind or DEFAULT_KIND[part],
            environment_id=own.environment_id or shared.environment_id or "",
            hardware_tier_id=own.hardware_tier_id or shared.hardware_tier_id or "",
            autoscaling=own.autoscaling or shared.autoscaling,
            settings=dict(sorted({**shared.settings, **own.settings}.items())),
        )

    def wait_limits(self) -> WaitLimits:
        """How long to wait for an App to run, and how often to look."""
        return WaitLimits(
            self.wait_timeout_s, self.poll_interval_s, self.no_instance_grace_s
        )


def refuse_secrets(settings: DeploySettings) -> None:
    """Raise when a ``settings`` block holds a secret: it would land in a script.

    Example::

        refuse_secrets(DeploySettings())   # DominoDeployError("RAG_ES_API_KEY ...")
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
    """Raise when a value the deploy of ``parts`` uses still holds a placeholder.

    Example::

        refuse_placeholders(DeploySettings(), ["workflow"])   # "... '<dataset>' ..."
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
        yield "hardware_tier_id", name, block.hardware_tier_id or ""
        for key, value in block.settings.items():
            yield key, f"{name}.settings", value


def _secrets_in(values: dict[str, str]) -> list[str]:
    # RAG_OVMS_AUTH is a secret only when it holds a token ("domino" names a mode).
    secrets: list[str] = [
        n for n in ("RAG_ES_API_KEY", "OPENAI_API_KEY") if n in values
    ]
    if values.get("RAG_OVMS_AUTH") not in (None, "", "domino"):
        secrets.append("RAG_OVMS_AUTH")
    return secrets
