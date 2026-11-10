"""Helpers and constants shared by the rag-deploy tests over the fake Domino.

Imported by ``test_domino_apps.py``, ``test_domino_deploy.py`` and
``test_rag_deploy_cli.py``.
"""

from __future__ import annotations

from rag_load_test.deploy.apps_client import DominoAppsClient
from rag_load_test.deploy.apps_reference import (
    AppCreationRequest,
    AppVersionCreationRequest,
)
from rag_load_test.deploy.fake_domino import (
    FAKE_DOMINO_URL,
    FakeDominoApps,
    create_fake_domino,
    fake_domino_http,
)
from rag_load_test.deploy.settings import DeploySettings

APPS_PATH = "/api/apps/beta/apps"
APPS_URL = f"{FAKE_DOMINO_URL}/apps"  # the fake's App URLs: {APPS_URL}/app-1, ...
LAUNCH = "/repos/rag topology/domino/app.sh"  # a space, to check the quoting
BOOTSTRAP_AND_EXEC = (
    "export RAG_BOOTSTRAP=1\nexec bash '/repos/rag topology/domino/app.sh'"
)


def add_app(
    apps: FakeDominoApps, name: str, project_id: str = "proj-1", script: str = ""
) -> None:
    """An App that exists before the test runs (not recorded as a create)."""
    version = AppVersionCreationRequest(entryScript=script or None)
    request = AppCreationRequest(
        name=name, projectId=project_id, visibility="AUTHENTICATED", version=version
    )
    apps.add_app(request)


def make_client(apps: FakeDominoApps, project_id: str = "proj-1") -> DominoAppsClient:
    return DominoAppsClient(fake_domino_http(create_fake_domino(apps)), project_id)


def no_sleep(seconds: float) -> None:
    pass


def make_settings(**overrides: object) -> DeploySettings:
    """DeploySettings without reading a stray ``deploy.env``."""
    values: dict[str, object] = {
        "project_id": "proj-1",
        "environment_id": "env-1",
        **overrides,
    }
    return DeploySettings(_env_file=None, **values)  # type: ignore[call-arg]


DEPLOY_ENV = """\
RAG_DEPLOY_TOPOLOGY=split-reranker
RAG_RERANKER_MODEL=/mnt/data/models/bge reranker
RAG_ES_URL=https://es.test:9200
PATH=/should/not/pass
OPENAI_BASE_URL=https://gateway.test
RAG_EMBEDDER_MODEL=/mnt/data/models/bge-small
"""
