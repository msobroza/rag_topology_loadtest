"""Helpers and constants shared by the rag-deploy tests over the fake Domino.

Imported by the ``test_domino_*.py`` files and ``test_rag_deploy_cli.py``. The
CLI tests' fixtures are in ``domino_deploy_fixtures.py``.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

from fake_domino.api_shapes import (
    AppCreationRequest,
    AppVersionCreationRequest,
    ConfigurationType,
)
from fake_domino.server import FAKE_DOMINO_URL, create_fake_domino, fake_domino_http
from fake_domino.state import FakeDominoApps
from rag_load_test.deploy.app_id_cache import AppIdCache
from rag_load_test.deploy.apps_client import DominoAppsClient
from rag_load_test.deploy.settings import DeploySettings

APPS_PATH = "/api/apps/beta/apps"
APPS_URL = f"{FAKE_DOMINO_URL}/apps"  # the fake's App pages: {APPS_URL}/app-1, ...
# Where the fake's App instances answer: {INSTANCES_URL}/app-1-v1-i, ...
INSTANCES_URL = f"{FAKE_DOMINO_URL}/fake-owner/fake-project/r/notebookSession"
LAUNCH = "/repos/rag topology/domino/app.sh"  # a space, to check the quoting
BOOTSTRAP_AND_EXEC = (
    "export RAG_BOOTSTRAP=1\nexec bash '/repos/rag topology/domino/app.sh'"
)
# What an Agent (the workflow App by default) exports before RAG_BOOTSTRAP.
AGENT_EXPORTS = "export RAG_DOMINO_TRACING=1\nexport RAG_BOOTSTRAP_EXTRAS=domino\n"


def add_app(
    apps: FakeDominoApps,
    name: str,
    project_id: str = "proj-1",
    script: str = "",
    configuration_type: ConfigurationType = "STANDARD",
) -> str:
    """An App that exists before the test runs (not recorded as a create); its id."""
    version = AppVersionCreationRequest(entryScript=script or None)
    request = AppCreationRequest(
        name=name,
        projectId=project_id,
        visibility="AUTHENTICATED",
        version=version,
        configurationType=configuration_type,
    )
    return apps.add_app(request).id


def make_client(
    apps: FakeDominoApps, project_id: str = "proj-1", cache: AppIdCache | None = None
) -> DominoAppsClient:
    http = fake_domino_http(create_fake_domino(apps))
    return DominoAppsClient(http, project_id, cache or AppIdCache(None))


def no_sleep(_seconds: float) -> None:
    pass


def make_settings(**overrides: object) -> DeploySettings:
    """DeploySettings with every App on ``env-1``; a stray deploy.yaml cannot leak in.

    Every key the tests depend on is passed, and passed values beat the file.
    """
    values: dict[str, object] = {
        "topology": "split-all",
        "model_server": "fastapi",
        "project_id": "proj-1",
        "ovms_bundle": "",
        "domino_host": "",
        "defaults": {"environment_id": "env-1"},
        "apps": {},
        **overrides,
    }
    return DeploySettings(**values)  # type: ignore[arg-type]


# The owner's example, with a split-reranker topology so it deploys two Apps.
DEPLOY_YAML = """\
topology: split-reranker
model_server: fastapi
project_id: ""

defaults:
  environment_id: env-123
  hardware_tier_id: small
  settings:
    RAG_ES_URL: https://es.test:9200
    RAG_RERANKER_MODEL: /mnt/data/models/bge reranker

apps:
  workflow:
    kind: agent
    autoscaling: {min_replicas: 1, max_replicas: 4}
  embedder:
    kind: app
    hardware_tier_id: cpu-8
    settings:
      RAG_EMBEDDER_MODEL: /mnt/data/models/bge-small
  reranker:
    kind: app
    hardware_tier_id: cpu-16
    settings:
      RAG_MODEL_WORKERS: 2
"""


# --- rag-deploy end to end: main over the fake Domino (fixtures and helpers) ---

PROXY = "http://proxy.test:8899"
# The CLI tests' deploy.yaml: every App on env-1, nothing else set.
ENV_1 = "defaults:\n  environment_id: env-1\n"
CACHE_FILE = ".rag-deploy-cache.json"  # in the working directory
NO_ENVIRONMENT = (
    "rag-deploy: error: No environment_id for App rag-embedder: set "
    "defaults.environment_id (or apps.embedder.environment_id) in deploy.yaml; it "
    "is the id of the Domino compute environment the App runs in\n"
)


def launched() -> str:
    """The tail of every entry script when rag-deploy runs from this directory."""
    return f"exec bash {shlex.quote(str(Path.cwd() / 'domino' / 'app.sh'))}"


RunMain = Callable[..., tuple[int, str, str]]


def deployed_lines(count: int) -> str:
    names = ["rag-embedder", "rag-reranker", "rag-workflow"][:count]
    return "".join(
        f"{name}: Running {INSTANCES_URL}/app-{n}-v1-i\n"
        for n, name in enumerate(names, 1)
    )
