"""rag-deploy's settings: ``deploy.yaml``, overridden by ``RAG_DEPLOY_*`` variables.

Top-level keys say what to deploy and how ``rag-deploy`` talks to Domino;
``defaults`` and ``apps.<part>`` say how each App runs. ``deploy.example.yaml``
documents every key.

Example:
    >>> DeploySettings().for_app("workflow").kind
    'agent'
    >>> DeploySettings().for_app("reranker").server
    'fastapi'
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

from ..domain.api_endpoint import DOMINO_RUN_TOKEN
from ..domain.roles import WORKFLOW_APP, AppPart
from .apps_client import DominoDeployError, WaitLimits
from .model_server_kinds.base_model_server_kind import MODEL_SERVER_KINDS

AppKind = Literal["app", "agent"]
# Where a topology alias puts a component it names: in the workflow App's
# process, or behind a model server App of its own. Each is also the backend
# the workflow App is told for the component.
Placement = Literal["local", "served"]
SERVED: Placement = "served"
DEPLOY_FILE = (
    "deploy.yaml"  # in the working directory, unless RAG_DEPLOY_FILE names one
)
# What serves a model server App whose block and the defaults name no kind.
DEFAULT_MODEL_SERVER_KIND = "fastapi"


def deploy_file() -> str:
    """Name the settings file: ``RAG_DEPLOY_FILE``, else ``deploy.yaml`` here.

    Returns:
        The path of the settings file.

    Example:
        >>> deploy_file()
        'deploy.yaml'
    """
    return os.environ.get("RAG_DEPLOY_FILE") or DEPLOY_FILE


class _Block(BaseModel):
    # A misspelt key must be reported, not silently ignored.
    model_config = ConfigDict(extra="forbid")


_PERCENTS = range(1, 101)  # what a CPU utilisation target can be


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
        if self.target_cpu_pct is not None and self.target_cpu_pct not in _PERCENTS:
            raise ValueError(f"target_cpu_pct {self.target_cpu_pct} is outside 1..100")
        return self


class AppBlock(_Block):
    """``defaults`` or one ``apps.<part>`` entry; None = not set here."""

    kind: AppKind | None = None
    environment_id: str | None = None
    environment_revision_id: str | None = None  # one build of the environment
    hardware_tier_id: str | None = None
    autoscaling: Autoscaling | None = None
    # The model-server kind of a model server App: a registered kind's name.
    server: str | None = None
    settings: dict[str, str] = {}

    @field_validator("server")
    @classmethod
    def _a_registered_kind(cls, value: str | None) -> str | None:
        if value is not None:
            MODEL_SERVER_KINDS.get(value)  # names every kind when it is none
        return value

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
    name: str  # in Domino: <app_name_prefix>-<part>
    kind: AppKind
    environment_id: str
    environment_revision_id: str  # "" = none: Domino chooses the revision
    hardware_tier_id: str
    autoscaling: Autoscaling | None
    server: str  # the model-server kind, which only a model server App uses
    settings: dict[str, str] = field(default_factory=dict)


# Our rule, not Domino's: the Apps API reference this tool follows gives an App's
# name as a plain string, and we found no rule of Domino's for it. So the prefix
# keeps to a conservative shape: lowercase words of letters and digits joined by
# single hyphens, starting with a letter. It never makes "-workflow", "rag--workflow"
# or a name with a space, a slash or a capital.
_APP_NAME_PREFIX = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


class DeploySettings(BaseSettings):
    """``deploy.yaml``; ``RAG_DEPLOY_<KEY>`` overrides a top-level key."""

    model_config = SettingsConfigDict(env_prefix="RAG_DEPLOY_", extra="forbid")

    # Each App is named <app_name_prefix>-<part>, so that two use cases with two
    # prefixes can share one Domino project.
    app_name_prefix: str = "rag"
    # The topology alias deployed: a key of topology_alias, checked against the
    # workflow file before anything is published (topology_alias.py).
    topology: str = "split-all"
    # Each topology alias: local or served for each component of the workflow
    # file it names. A component it does not name keeps the backend the
    # workflow file gives it. Where a component runs differs per site, so no
    # alias is defined in code (ADR 0003).
    topology_alias: dict[str, dict[str, Placement]] = {}
    project_id: str = ""  # "" = DOMINO_PROJECT_ID, which Domino sets in a Workspace
    ovms_bundle: str = ""  # required by an ovms App: bin/ovms + lib/, or its .tar.gz
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
    # Keyed by workflow or by a served component's name: any other key is
    # refused once the workflow file is read (topology_alias.py).
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
        """Order the sources: init kwargs > environment > deploy.yaml.

        Args:
            settings_cls: The settings class being built.
            init_settings: Constructor arguments.
            env_settings: ``RAG_DEPLOY_*`` environment variables.
            dotenv_settings: Not read: deploy settings take no ``.env`` file.
            file_secret_settings: Not read: nor a secrets directory.

        Returns:
            The sources, highest priority first.
        """
        # Pydantic passes every source by keyword, so the names stay.
        del dotenv_settings, file_secret_settings
        yaml_file = YamlConfigSettingsSource(settings_cls, yaml_file=deploy_file())
        return init_settings, env_settings, yaml_file

    @field_validator("app_name_prefix")
    @classmethod
    def _a_valid_app_name_prefix(cls, value: str) -> str:
        if _APP_NAME_PREFIX.fullmatch(value):
            return value
        raise ValueError(
            f"app_name_prefix {value!r} would not make a valid App name: it must be "
            "lowercase letters and digits in words joined by single hyphens, "
            "starting with a letter, such as 'rag' or 'qp-rag' (rag-deploy's rule, "
            "not Domino's)"
        )

    @model_validator(mode="after")
    def _no_kind_for_the_workflow_app(self) -> DeploySettings:
        # The workflow App is no model server: a kind in its block is a mistake
        # that would otherwise be silently unused.
        workflow = self.apps.get(WORKFLOW_APP, AppBlock())
        if workflow.server is None:
            return self
        raise ValueError(
            f"apps.{WORKFLOW_APP}.server is {workflow.server!r}, but the workflow App "
            "is no model server: set server in the block of a served component, "
            "or in defaults for every model server App"
        )

    @model_validator(mode="after")
    def _project_from_the_workspace(self) -> DeploySettings:
        # The example's `project_id: ""` means "not set", not "".
        self.project_id = self.project_id or os.environ.get("DOMINO_PROJECT_ID", "")
        return self

    def app_name(self, part: AppPart) -> str:
        """Name ``part``'s App in Domino: ``<app_name_prefix>-<part>``.

        Args:
            part: The App part.

        Returns:
            The App's exact name, by which it is created and found.

        Example:
            >>> DeploySettings(app_name_prefix="qp-rag").app_name("workflow")
            'qp-rag-workflow'
        """
        return f"{self.app_name_prefix}-{part}"

    def for_app(self, part: AppPart) -> AppSettings:
        """Merge ``part``'s App settings: its own value, else the default.

        ``settings`` entries are merged key by key, the App's own winning. The
        default environment revision goes only with the default environment: an
        App that sets its own ``environment_id`` does not take it.

        Args:
            part: The App whose settings are merged.

        Returns:
            The App's settings, the kind defaulting to ``agent`` for the
            workflow App and ``app`` for a model server App, and the
            model-server kind to ``fastapi``.

        Example:
            >>> DeploySettings().for_app("reranker").hardware_tier_id
            'cpu-16'
        """
        own, shared = self.apps.get(part, AppBlock()), self.defaults
        environment_id, revision_id = _environment_of(own, shared)
        return AppSettings(
            part=part,
            name=self.app_name(part),
            kind=own.kind or shared.kind or _default_kind(part),
            environment_id=environment_id,
            environment_revision_id=revision_id,
            hardware_tier_id=own.hardware_tier_id or shared.hardware_tier_id or "",
            autoscaling=own.autoscaling or shared.autoscaling,
            server=own.server or shared.server or DEFAULT_MODEL_SERVER_KIND,
            settings=dict(sorted({**shared.settings, **own.settings}.items())),
        )

    def wait_limits(self) -> WaitLimits:
        """Say how long to wait for an App to run, and how often to look.

        Returns:
            The ``wait_timeout_s``, ``poll_interval_s`` and ``no_instance_grace_s``
            settings.
        """
        return WaitLimits(
            self.wait_timeout_s, self.poll_interval_s, self.no_instance_grace_s
        )


def _default_kind(part: AppPart) -> AppKind:
    # The orchestrator is published as an Agent, the model servers as plain Apps.
    return "agent" if part == WORKFLOW_APP else "app"


def _environment_of(own: AppBlock, shared: AppBlock) -> tuple[str, str]:
    # A revision is one build of one environment, so the default revision is a
    # build of the default environment: given to an App that names another
    # environment, it would pair that environment with a build of a different one.
    if own.environment_id:
        return own.environment_id, own.environment_revision_id or ""
    revision_id = own.environment_revision_id or shared.environment_revision_id
    return shared.environment_id or "", revision_id or ""


def refuse_secrets(settings: DeploySettings) -> None:
    """Refuse a ``settings`` block that holds a secret: it would land in a script.

    Args:
        settings: The deploy settings whose ``defaults`` and ``apps`` are checked.

    Raises:
        DominoDeployError: If a setting whose name ends with ``_API_KEY``
            (``RAG_COMPONENTS__LLM__API__API_KEY``, the vector store's key, ...)
            holds a value other than "" or ``domino``.

    Example:
        >>> refuse_secrets(DeploySettings())
        DominoDeployError: RAG_COMPONENTS__LLM__API__API_KEY ...
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
        for key, value in block.settings.items():
            yield key, f"{name}.settings", value


def _secrets_in(values: dict[str, str]) -> list[str]:
    # An API key is a secret unless it is empty or names the Domino run token.
    return [
        name
        for name, value in values.items()
        if name.endswith("_API_KEY") and value not in ("", DOMINO_RUN_TOKEN)
    ]
