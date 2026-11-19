"""What a listing reads of ``deploy.yaml``: how to reach Domino, and nothing else.

A listing only reads Domino, so no App block nor ``app_name_prefix`` needs to
be valid: the other keys are ignored here and checked by ``DeploySettings``
when a command needs them. The defaults are ``DeploySettings``'s own.

Example:
    >>> ListingSettings().http_retries
    2
"""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ..deploy_settings import DeploySettings, project_or_workspace

_DEPLOY = DeploySettings.model_fields  # one place for each default


class ListingSettings(BaseSettings):
    """The keys of deploy.yaml that reach Domino; QP_DEPLOY_<KEY> overrides one."""

    model_config = SettingsConfigDict(env_prefix="QP_DEPLOY_", extra="ignore")

    project_id: str = _DEPLOY["project_id"].default
    http_timeout_s: float = _DEPLOY["http_timeout_s"].default
    http_retries: int = _DEPLOY["http_retries"].default

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Take the sources ``DeploySettings`` takes: init > environment > the file.

        Args:
            settings_cls: The settings class being built.
            init_settings: Constructor arguments.
            env_settings: ``QP_DEPLOY_*`` environment variables.
            dotenv_settings: Not read.
            file_secret_settings: Not read.

        Returns:
            The sources, highest priority first.
        """
        return DeploySettings.settings_customise_sources(
            settings_cls,
            init_settings=init_settings,
            env_settings=env_settings,
            dotenv_settings=dotenv_settings,
            file_secret_settings=file_secret_settings,
        )

    @model_validator(mode="after")
    def _project_from_the_workspace(self) -> ListingSettings:
        self.project_id = project_or_workspace(self.project_id)
        return self
