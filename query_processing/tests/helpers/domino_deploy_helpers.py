"""Helpers and constants shared by the qp-deploy tests over the fake Domino.

Imported by the ``test_domino_*.py`` files and ``test_qp_deploy.py``. The
CLI tests' fixtures are in ``domino_deploy_fixtures.py``.
"""

from __future__ import annotations

import shlex
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx

from fake_domino.api_shapes import (
    AppCreationRequest,
    AppVersionCreationRequest,
    ConfigurationType,
)
from fake_domino.server import FAKE_DOMINO_URL, create_fake_domino, fake_domino_http
from fake_domino.stand_ins import fake_environment
from fake_domino.state import FakeDominoApps
from launch_script_stand_ins import DOCLANG_WHEEL
from query_processing.deploy.app_id_cache import AppIdCache
from query_processing.deploy.apps_client import DominoAppsClient
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.publish import LaunchPaths

APPS_PATH = "/api/apps/beta/apps"
APPS_URL = f"{FAKE_DOMINO_URL}/apps"  # the fake's App pages: {APPS_URL}/app-1, ...
# Where the fake's App instances answer: {INSTANCES_URL}/app-1-v1-i, ...
INSTANCES_URL = f"{FAKE_DOMINO_URL}/fake-owner/fake-project/r/notebookSession"
# The toolkit's launch script and a use case beside it, in a repository whose
# path has a space, to check the quoting.
LAUNCH = LaunchPaths(
    "/repos/rag topology/query_processing/domino/app.sh",
    "/repos/rag topology/my_use_case",
)


def a_wheels_directory_on_the_dataset() -> str:
    """The directory of wheels every deploy file of the tests names, made here.

    A deploy refuses a directory that does not exist on the Workspace or holds
    not exactly one wheel of doclang's core, so it is a real one, holding a
    stand-in wheel (an empty file under the name a build gives it). It sits in
    the system's temporary folder under a fixed name, so that every constant
    below can name it at import; making it again changes nothing.

    Example::

        WHEELS = a_wheels_directory_on_the_dataset()  # ".../qp-deploy-tests/wheels"
    """
    wheels = Path(tempfile.gettempdir()) / "qp-deploy-tests" / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    (wheels / DOCLANG_WHEEL).touch()
    return str(wheels)


# Where the Apps that install find doclang's core: a directory on the Dataset,
# which every deploy file the tests write names.
WHEELS = a_wheels_directory_on_the_dataset()
WHEELS_YAML = f"bootstrap_wheels: '{WHEELS}'\n"
# How every entry script ends: the use case's directory, the bootstrap and its
# wheels, the script.
BOOTSTRAP_AND_EXEC = (
    "export QP_USE_CASE_DIR='/repos/rag topology/my_use_case'\n"
    "export QP_BOOTSTRAP=1\n"
    f"export QP_BOOTSTRAP_WHEELS={shlex.quote(WHEELS)}\n"
    "exec bash '/repos/rag topology/query_processing/domino/app.sh'"
)
# What an Agent (the workflow App by default) exports before QP_BOOTSTRAP.
AGENT_EXPORTS = "export QP_DOMINO__TRACING=1\nexport QP_BOOTSTRAP_EXTRAS=domino\n"


def model_server_exports(part: str, kind: str = "fastapi") -> str:
    """What the model server App of ``part`` served by ``kind`` exports first.

    Its Role, its component and its model-server kind, before what the kind adds.
    """
    return (
        "export QP_ROLE=model-server\n"
        f"export QP_MODEL_SERVER_COMPONENT={part}\n"
        f"export QP_MODEL_SERVER_KIND={kind}\n"
    )


# What the monolith alias's workflow App exports after its Role and topology:
# the backend of each component the alias names, and the run label.
MONOLITH_EXPORTS = (
    "export QP_COMPONENTS__EMBEDDER__BACKEND=local\n"
    "export QP_COMPONENTS__RERANKER__BACKEND=local\n"
    "export QP_RUN_LABEL=monolith\n"
)


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


def http_on(apps: FakeDominoApps) -> httpx.Client:
    """A client whose Domino is the in-process fake holding ``apps``."""
    return fake_domino_http(create_fake_domino(apps))


# --- the lookups and the listings: what the fake Domino has ---

# When the fake's revisions were built, in UTC.
BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
# Where a refusal says the value was set.
WHERE = "App rag-reranker (apps.reranker or defaults in deploy.yaml)"
# The standard environment: qp-rag, revisions 7 (latest), 6 (active) and 5, the
# last two built the same day. Not ENV_1, which is the text of a deploy file.
FAKE_ENV_1 = fake_environment(
    "env-1",
    (7, "rev-7", BUILT_7),
    (6, "rev-6", BUILT_6),
    (5, "rev-5", BUILT_6),
    name="qp-rag",
    active=6,
)


def make_settings(**given: object) -> DeploySettings:
    """DeploySettings with every App on ``env-1``; a stray deploy.yaml cannot leak in.

    Every key the tests depend on is passed, and passed values beat the file.
    The topology aliases are the three of deploy.example.yaml; no overrides
    unless ``overrides`` is given.
    """
    values: dict[str, object] = {
        "app_name_prefix": "rag",
        "topology": "split-all",
        "topology_alias": TOPOLOGY_ALIASES,
        "project_id": "proj-1",
        "ovms_bundle": "",
        "bootstrap_wheels": WHEELS,
        "domino_host": "",
        "defaults": {"environment_id": "env-1"},
        "apps": {},
        "overrides": {},
        **given,
    }
    return DeploySettings(**values)  # type: ignore[arg-type]


# The tests' Apps are named rag-<part>. A deploy file must name its prefix: the
# toolkit has no default for it.
PREFIX_YAML = "app_name_prefix: rag\n"
# The three topology aliases compared today, as deploy.example.yaml defines them.
TOPOLOGY_ALIASES = {
    "monolith": {"embedder": "local", "reranker": "local"},
    "split-reranker": {"embedder": "local", "reranker": "served"},
    "split-all": {"embedder": "served", "reranker": "served"},
}
TOPOLOGY_ALIAS_YAML = """\
topology_alias:
  monolith: {embedder: local, reranker: local}
  split-reranker: {embedder: local, reranker: served}
  split-all: {embedder: served, reranker: served}
"""

# The owner's example, with a split-reranker topology so it deploys two Apps. The
# free lists hold variables that are no value of the workflow file; the model
# paths are overrides, for every App.
DEPLOY_YAML = "topology: split-reranker\n" + PREFIX_YAML + WHEELS_YAML
DEPLOY_YAML += TOPOLOGY_ALIAS_YAML + """\
project_id: ""

defaults:
  environment_id: env-123
  hardware_tier_id: small
  server: fastapi
  settings:
    QP_ES_URL: https://es.test:9200

apps:
  workflow:
    kind: agent
    autoscaling: {min_replicas: 1, max_replicas: 4}
  embedder:
    kind: app
    hardware_tier_id: cpu-8
    settings:
      QP_BOOTSTRAP_EXTRAS: dev
  reranker:
    kind: app
    hardware_tier_id: cpu-16
    settings:
      QP_MODEL_WORKERS: 2

overrides:
  components:
    reranker: {source_id: /mnt/data/models/bge reranker}
    embedder: {source_id: /mnt/data/models/bge-small}
"""
# What every App of DEPLOY_YAML exports after its free list: the overrides.
DEPLOY_YAML_OVERRIDES = (
    "export QP_COMPONENTS__RERANKER__SOURCE_ID='/mnt/data/models/bge reranker'\n"
    "export QP_COMPONENTS__EMBEDDER__SOURCE_ID=/mnt/data/models/bge-small\n"
)


# --- qp-deploy end to end: main over the fake Domino (fixtures and helpers) ---

PROXY = "http://proxy.test:8899"
# The CLI tests' deploy.yaml: the prefix, the wheels' directory, the three topology
# aliases and every App on env-1, nothing else set. It ends inside defaults, so a
# test can add a default to it.
ENV_1 = PREFIX_YAML + WHEELS_YAML + TOPOLOGY_ALIAS_YAML
ENV_1 += "defaults:\n  environment_id: env-1\n"
CACHE_FILE = ".qp-deploy-cache.json"  # in the working directory
NO_ENVIRONMENT = (
    "qp-deploy: error: No environment_id for App rag-embedder: set "
    "defaults.environment_id (or apps.embedder.environment_id) in deploy.yaml; it "
    "is the id of the Domino compute environment the App runs in\n"
)


def launched() -> str:
    """The tail of every entry script when qp-deploy runs from this directory.

    This directory is the use case's, beside the toolkit's in the repository,
    as ``domino_env`` lays them out: its path, the bootstrap and its wheels,
    then the toolkit's launch script.
    """
    use_case = Path.cwd()
    script = use_case.parent / "query_processing" / "domino" / "app.sh"
    return (
        f"export QP_USE_CASE_DIR={shlex.quote(str(use_case))}\n"
        "export QP_BOOTSTRAP=1\n"
        f"export QP_BOOTSTRAP_WHEELS={shlex.quote(WHEELS)}\n"
        f"exec bash {shlex.quote(str(script))}"
    )


RunMain = Callable[..., tuple[int, str, str]]


def deployed_lines(count: int) -> str:
    names = ["rag-embedder", "rag-reranker", "rag-workflow"][:count]
    return "".join(
        f"{name}: Running {INSTANCES_URL}/app-{n}-v1-i\n"
        for n, name in enumerate(names, 1)
    )
