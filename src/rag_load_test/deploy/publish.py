"""Publishing a topology's Apps: their entry scripts, versions and order.

The model server Apps are published first, then the workflow App with their URLs.
"""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Callable, Mapping
from typing import Literal

from ..domain.topology import Topology
from .apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    wait_until_running,
)
from .settings import AppPart, DeploySettings

LAUNCH_SCRIPT = "domino/app.sh"  # relative to the repository root

APPS_BY_TOPOLOGY: dict[Topology, list[AppPart]] = {
    "split-all": ["embedder", "reranker", "workflow"],
    "split-reranker": ["reranker", "workflow"],
    "monolith": ["workflow"],
}


def apps_for_topology(topology: Topology) -> list[AppPart]:
    """The Apps ``topology`` needs, model servers first.

    Example::

        apps_for_topology("split-reranker")   # ["reranker", "workflow"]
    """
    return list(APPS_BY_TOPOLOGY[topology])


def launch_script_in(directory: str) -> str:
    """The absolute path of ``domino/app.sh`` under ``directory``, or raise.

    The Apps mount this repository at the same path as the Workspace (for example
    /repos/<project>) but do not start in it, so the path must be absolute.

    Example::

        launch_script_in(os.getcwd())   # "/repos/rag/domino/app.sh"
    """
    path = os.path.join(directory, LAUNCH_SCRIPT)
    if not os.path.isfile(path):
        raise DominoDeployError(
            f"{directory} has no {LAUNCH_SCRIPT}: run rag-deploy from the repository "
            "root (for example /repos/<project>), the directory the Apps run it from"
        )
    return path


def entry_script(
    part: AppPart,
    settings: DeploySettings,
    urls: Mapping[AppPart, str],
    app_settings: Mapping[str, str],
    launch_script: str,
) -> str:
    """Shell that exports one App's settings, then runs ``launch_script``.

    Domino has no per-App environment variables, so each App carries its own:
    ``app_settings`` (from ``deploy.env``) first, then the computed ones, which
    therefore win. ``urls`` holds the model server Apps' URLs (workflow App only).

    Example::

        entry_script("reranker", DeploySettings(), {}, {}, "/repos/rag/domino/app.sh")
        # "export RAG_ROLE=fastapi-reranker\n...\nexec bash /repos/rag/domino/app.sh"
    """
    if part == "workflow":
        computed = _workflow_exports(settings, urls, app_settings)
    else:
        computed = _model_server_exports(part, settings)
    # The generic environment has nothing installed: app.sh installs it first.
    exports = [*app_settings.items(), *computed.items(), ("RAG_BOOTSTRAP", "1")]
    lines = [f"export {name}={shlex.quote(value)}" for name, value in exports]
    return "\n".join([*lines, f"exec bash {shlex.quote(launch_script)}"])


def _model_server_exports(part: AppPart, settings: DeploySettings) -> dict[str, str]:
    exports = {"RAG_ROLE": f"{settings.model_server}-{part}"}
    if settings.model_server == "ovms":
        exports["RAG_OVMS_BUNDLE"] = settings.ovms_bundle
    return exports


def _workflow_exports(
    settings: DeploySettings,
    urls: Mapping[AppPart, str],
    app_settings: Mapping[str, str],
) -> dict[str, str]:
    topology = settings.topology
    if topology == "monolith":
        return {"RAG_ROLE": "monolith", "RAG_TOPOLOGY": topology}
    exports = {"RAG_ROLE": "workflow", "RAG_TOPOLOGY": topology}
    exports["RAG_OVMS_RERANK_URL"] = urls["reranker"]
    if topology == "split-all":
        exports["RAG_OVMS_EMBEDDINGS_URL"] = urls["embedder"]
    if "RAG_RUN_LABEL" not in app_settings:
        exports["RAG_RUN_LABEL"] = f"{topology}-{settings.model_server}"
    return exports


def version_body(settings: DeploySettings, script: str) -> dict[str, str]:
    """The version object: ``entryScript``, the environment and the tier when set.

    Example::

        version_body(DeploySettings(environment_id="e1"), "exec bash domino/app.sh")
        # {"entryScript": "exec bash domino/app.sh", "environmentId": "e1"}
    """
    body = {"entryScript": script, "environmentId": settings.environment_id}
    if settings.hardware_tier_id:
        body["hardwareTierId"] = settings.hardware_tier_id
    return body


def publish_app(
    client: DominoAppsClient,
    settings: DeploySettings,
    part: AppPart,
    urls: Mapping[AppPart, str],
    app_settings: Mapping[str, str],
    *,
    launch_script: str,
    sleep: Callable[[float], None] = time.sleep,
) -> DominoApp:
    """Create ``rag-<part>`` or post a new version to it, then wait until Running.

    Example::

        publish_app(client, DeploySettings(), "reranker", {}, {}, launch_script=path)
    """
    name = f"rag-{part}"
    script = entry_script(part, settings, urls, app_settings, launch_script)
    version = version_body(settings, script)
    app = client.find(name)
    if app is None:
        app = client.create(name, version)
        version_id = app.version_id
    else:
        version_id = client.add_version(app.id, version)
    return wait_until_running(client, app, version_id, sleep=sleep)


def deploy(
    client: DominoAppsClient,
    settings: DeploySettings,
    target: AppPart | Literal["all"],
    app_settings: Mapping[str, str],
    *,
    launch_script: str,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Publish ``target`` (or every App the topology needs, in order).

    Example::

        deploy(client, DeploySettings(), "all", {}, launch_script=path)
    """
    needed = apps_for_topology(settings.topology)
    if target != "all" and target not in needed:
        names = ", ".join(f"rag-{part}" for part in needed)
        raise DominoDeployError(
            f"the {settings.topology} topology does not use App rag-{target}; "
            f"it needs: {names}"
        )
    parts = needed if target == "all" else [target]
    _check_deploy_settings(settings, parts)
    for part in parts:
        urls = model_server_urls(client, settings) if part == "workflow" else {}
        publish_app(
            client,
            settings,
            part,
            urls,
            app_settings,
            launch_script=launch_script,
            sleep=sleep,
        )


def _check_deploy_settings(settings: DeploySettings, parts: list[AppPart]) -> None:
    if not settings.environment_id:
        raise DominoDeployError(
            "RAG_DEPLOY_ENVIRONMENT_ID is not set: it is the id of the Domino "
            "compute environment the Apps run in"
        )
    if settings.model_server != "ovms" or parts == ["workflow"]:
        return
    if not settings.ovms_bundle or not os.path.exists(settings.ovms_bundle):
        raise DominoDeployError(
            f"RAG_DEPLOY_OVMS_BUNDLE={settings.ovms_bundle} does not exist: set it to "
            "the OVMS bundle folder (bin/ovms, lib/) or its .tar.gz, at a path the "
            "Apps can read"
        )


def model_server_urls(
    client: DominoAppsClient, settings: DeploySettings
) -> dict[AppPart, str]:
    """The URLs of the model server Apps the workflow App calls, or raise.

    Example::

        model_server_urls(client, DeploySettings())   # {"embedder": ..., ...}
    """
    model_servers = apps_for_topology(settings.topology)[:-1]  # workflow is last
    return {part: require_app(client, part).url for part in model_servers}


def require_app(client: DominoAppsClient, part: AppPart) -> DominoApp:
    """The deployed ``rag-<part>`` App, or an error naming the make target.

    Example::

        require_app(client, "workflow").url   # "https://domino.example/apps/a1"
    """
    app = client.find(f"rag-{part}")
    if app is None:
        raise DominoDeployError(
            f"App rag-{part} is not deployed in project {client.project_id}: "
            f"run make deploy-{part}"
        )
    return app
