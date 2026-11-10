"""rag-deploy's settings: ``RAG_DEPLOY_*``, and the Apps' ``RAG_*`` from deploy.env."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from dotenv import dotenv_values
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..domain.topology import Topology
from .apps_client import DominoDeployError

AppPart = Literal["embedder", "reranker", "workflow"]
ModelServer = Literal["fastapi", "ovms"]
DEPLOY_ENV_FILE = "deploy.env"


class DeploySettings(BaseSettings):
    """``RAG_DEPLOY_*`` from ``deploy.env`` or the environment; "" = not set."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_DEPLOY_", env_file=DEPLOY_ENV_FILE, extra="ignore"
    )

    topology: Topology = "split-all"
    model_server: ModelServer = "fastapi"
    environment_id: str = ""  # required by ``rag-deploy deploy``
    hardware_tier_id: str = ""
    ovms_bundle: str = ""  # required for ``ovms``: bin/ovms + lib/, or its .tar.gz
    project_id: str = ""  # "" = DOMINO_PROJECT_ID, which Domino sets in a Workspace

    @model_validator(mode="after")
    def _project_from_the_workspace(self) -> DeploySettings:
        # The example's empty RAG_DEPLOY_PROJECT_ID= line means "not set", not "".
        self.project_id = self.project_id or os.environ.get("DOMINO_PROJECT_ID", "")
        return self


def load_app_settings(path: str = DEPLOY_ENV_FILE) -> dict[str, str]:
    """The ``RAG_*`` lines of ``path`` other than ``RAG_DEPLOY_*``, sorted by name.

    Read from the file only, never the environment: every App gets them as is.
    A missing file means none.

    Example::

        load_app_settings()   # {"RAG_ES_URL": "https://es.example:9200"}
    """
    values = dotenv_values(path)
    for secret in _secrets_in(values):
        raise DominoDeployError(
            f"{secret} is set in {path}, whose values are written into the "
            "Apps' launch scripts: set it as a Domino project or user environment "
            "variable instead"
        )
    return {
        name: value or ""
        for name, value in sorted(values.items())
        if name.startswith("RAG_") and not name.startswith("RAG_DEPLOY_")
    }


def _secrets_in(values: Mapping[str, str | None]) -> list[str]:
    # RAG_OVMS_AUTH is a secret only when it holds a token ("domino" names a mode).
    secrets = ["RAG_ES_API_KEY"] if "RAG_ES_API_KEY" in values else []
    if values.get("RAG_OVMS_AUTH") not in (None, "", "domino"):
        secrets.append("RAG_OVMS_AUTH")
    return secrets
