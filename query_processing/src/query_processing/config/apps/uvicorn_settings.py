"""What a process of ``qp-serve`` or ``qp-model-server`` runs as: Uvicorn's settings.

An App is one container on Kubernetes, scaled by replicas, so these are the
App's, not the workflow's: flat ``QP_UVICORN_*`` variables, set in the App's own
block of ``deploy.yaml`` (``apps.<component>.settings``) or in ``.env``, as the
``QP_OVMS_*`` ones are. Each default is Uvicorn's own (0.54.0), so an App that
sets nothing runs as it always has. The workers count is always passed:
Uvicorn otherwise reads ``WEB_CONCURRENCY``, and above 1 it exits on an app
object (``infrastructure/http/uvicorn_run.py`` passes the factory instead).

Example:
    >>> UvicornSettings().run_arguments()["workers"]
    1
    >>> UvicornSettings.variable_of("backlog")
    'QP_UVICORN_BACKLOG'
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .base_app_settings import BaseAppSettings

ENV_PREFIX = "QP_UVICORN_"


class UvicornSettings(BaseAppSettings):
    """Every ``QP_UVICORN_*`` variable, one field each.

    Example:
        >>> UvicornSettings(workers=2).run_arguments()["workers"]
        2
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=".env", extra="ignore"
    )

    # Processes serving the app, each loading its own models and planning its
    # threads on its share of the CPUs. FastAPI's docs: on Kubernetes, one
    # process a container and more replicas.
    workers: int = Field(default=1, ge=1)
    # Connections and tasks at once before Uvicorn answers 503; 0 = no limit.
    # This project queues overload, which lost fewer requests (model-servers.md).
    limit_concurrency: int = Field(default=0, ge=0)
    # Connections the socket holds while the event loop is busy.
    backlog: int = Field(default=2048, ge=1)
    # Seconds an idle kept-alive connection stays open.
    timeout_keep_alive_s: int = Field(default=5, ge=0)

    def run_arguments(self) -> dict[str, int | None]:
        """Return the settings as ``uvicorn.run``'s keywords.

        Returns:
            ``workers``, ``limit_concurrency`` (``None`` for 0, no limit),
            ``backlog`` and ``timeout_keep_alive``.

        Example:
            >>> UvicornSettings(limit_concurrency=0).run_arguments()["backlog"]
            2048
        """
        return {
            "workers": self.workers,
            "limit_concurrency": self.limit_concurrency or None,
            "backlog": self.backlog,
            "timeout_keep_alive": self.timeout_keep_alive_s,
        }
