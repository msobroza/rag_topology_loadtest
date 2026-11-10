"""Publishing a topology's Apps: their entry scripts, versions and order.

The model server Apps are published first, then the workflow App with the URLs
their running instances answer on.
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
from .endpoint_url import endpoint_url
from .settings import (
    AppKind,
    AppPart,
    AppSettings,
    Autoscaling,
    DeploySettings,
    deploy_file,
    refuse_placeholders,
    refuse_secrets,
)

LAUNCH_SCRIPT = "domino/app.sh"  # relative to the repository root
# An Agent is an App whose configurationType is AISYSTEM: the only API-level
# difference (assumed to be what the UI's "Deploy Agent" sends).
CONFIGURATION_TYPES: dict[AppKind, str] = {"app": "STANDARD", "agent": "AISYSTEM"}
# Exported by an Agent unless its settings set them: Domino tracing, and the
# [domino] extra that brings its library.
AGENT_EXPORTS = {"RAG_DOMINO_TRACING": "1", "RAG_BOOTSTRAP_EXTRAS": "domino"}

APPS_BY_TOPOLOGY: dict[Topology, list[AppPart]] = {
    "split-all": ["embedder", "reranker", "workflow"],
    "split-reranker": ["reranker", "workflow"],
    "monolith": ["workflow"],
}


def apps_for_topology(topology: Topology) -> list[AppPart]:
    """List the Apps ``topology`` needs, model servers first.

    Args:
        topology: The topology being deployed.

    Returns:
        Its App parts, the workflow App last.

    Example:
        >>> apps_for_topology("split-reranker")
        ['reranker', 'workflow']
    """
    return list(APPS_BY_TOPOLOGY[topology])


def launch_script_in(directory: str) -> str:
    """Return the absolute path of ``domino/app.sh`` under ``directory``.

    The Apps mount this repository at the same path as the Workspace (for example
    /repos/<project>) but do not start in it, so the path must be absolute.

    Args:
        directory: The repository root, absolute.

    Returns:
        The launch script's path.

    Raises:
        DominoDeployError: If ``directory`` has no ``domino/app.sh``.

    Example:
        >>> launch_script_in(os.getcwd())
        '/repos/rag/domino/app.sh'
    """
    path = os.path.join(directory, LAUNCH_SCRIPT)
    if not os.path.isfile(path):
        raise DominoDeployError(
            f"{directory} has no {LAUNCH_SCRIPT}: run rag-deploy from the repository "
            "root (for example /repos/<project>), the directory the Apps run it from"
        )
    return path


def entry_script(
    app: AppSettings,
    settings: DeploySettings,
    urls: Mapping[AppPart, str],
    launch_script: str,
) -> str:
    r"""Write the shell that exports one App's settings, then runs ``launch_script``.

    Each App carries its own environment variables: ``app.settings`` first, then
    the computed ones, which therefore win.

    Args:
        app: The App's merged settings.
        settings: The whole deploy settings (topology, model server, bundle).
        urls: Where the model server Apps answer (workflow App only).
        launch_script: Absolute path of ``domino/app.sh``.

    Returns:
        The entry script, one ``export`` per line, then ``exec bash``.

    Example:
        >>> app = settings.for_app("reranker")
        >>> entry_script(app, settings, {}, "/repos/rag/domino/app.sh")
        'export RAG_ROLE=fastapi-reranker\n...\nexec bash /repos/rag/domino/app.sh'
    """
    if app.part == "workflow":
        computed = _workflow_exports(settings, urls, app.settings)
    else:
        computed = _model_server_exports(app.part, settings)
    if app.kind == "agent":
        computed |= {k: v for k, v in AGENT_EXPORTS.items() if k not in app.settings}
    # The generic environment has nothing installed: app.sh installs it first.
    exports = [*app.settings.items(), *computed.items(), ("RAG_BOOTSTRAP", "1")]
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


def version_body(app: AppSettings, script: str) -> dict[str, object]:
    """Build the version object: script, environment, and tier and autoscaling if set.

    Args:
        app: The App's merged settings.
        script: Its entry script.

    Returns:
        The ``AppVersionCreationRequest`` body.

    Example:
        >>> version_body(settings.for_app("workflow"), "exec bash domino/app.sh")
        {'entryScript': 'exec bash domino/app.sh', 'environmentId': 'e1', ...}
    """
    body: dict[str, object] = {
        "entryScript": script,
        "environmentId": app.environment_id,
    }
    if app.hardware_tier_id:
        body["hardwareTierId"] = app.hardware_tier_id
    if app.autoscaling is not None:
        body["autoscalingSpecification"] = _autoscaling_specification(app.autoscaling)
    return body


def _autoscaling_specification(scaling: Autoscaling) -> dict[str, object]:
    spec: dict[str, object] = {
        "enabled": True,
        "minReplicas": scaling.min_replicas,
        "maxReplicas": scaling.max_replicas,
    }
    if scaling.target_cpu_pct is not None:
        spec["targetCpuAvgUtilizationPct"] = scaling.target_cpu_pct
    return spec


def publish_app(
    client: DominoAppsClient,
    settings: DeploySettings,
    part: AppPart,
    urls: Mapping[AppPart, str],
    *,
    launch_script: str,
    sleep: Callable[[float], None] = time.sleep,
) -> DominoApp:
    """Create ``rag-<part>`` or post a new version to it, then wait until Running.

    Prints the App's endpoint once Running.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings.
        part: The App to publish.
        urls: Where the model server Apps answer (workflow App only).
        launch_script: Absolute path of ``domino/app.sh``.
        sleep: Waits between polls; tests and dry runs pass a no-op.

    Returns:
        The App, Running.

    Raises:
        DominoDeployError: If the existing App is of another kind (App or Agent),
            or it does not come up.

    Example:
        >>> publish_app(client, DeploySettings(), "reranker", {}, launch_script=path)
    """
    app_settings = settings.for_app(part)
    script = entry_script(app_settings, settings, urls, launch_script)
    version = version_body(app_settings, script)
    app = client.find(app_settings.name)
    if app is None:
        kind = CONFIGURATION_TYPES[app_settings.kind]
        app = client.create(app_settings.name, version, kind)
        version_id = app.version_id
    else:
        _refuse_another_kind(app, app_settings)
        version_id = client.add_version(app.id, version)
    app = wait_until_running(
        client, app, version_id, settings.wait_limits(), sleep=sleep
    )
    print(f"{app.name}: Running {endpoint_url(client, app, settings.domino_host)}")
    return app


def _refuse_another_kind(app: DominoApp, wanted: AppSettings) -> None:
    # Converting an App to an Agent (or back) is not offered by the Apps API.
    if app.configuration_type == CONFIGURATION_TYPES[wanted.kind]:
        return
    kinds = {ctype: kind for kind, ctype in CONFIGURATION_TYPES.items()}
    existing = kinds.get(app.configuration_type, "unknown")
    raise DominoDeployError(
        f"App {app.name} exists with configurationType {app.configuration_type} "
        f"(kind {existing}), but {deploy_file()} asks for kind {wanted.kind} "
        f"({CONFIGURATION_TYPES[wanted.kind]}): delete {app.name} in Domino to "
        f"create it again, or set apps.{wanted.part}.kind: {existing}"
    )


def deploy(
    client: DominoAppsClient,
    settings: DeploySettings,
    target: AppPart | Literal["all"],
    *,
    launch_script: str,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Publish ``target``, or every App the topology needs, in order.

    Settings holding secrets, placeholders, no environment id or a missing OVMS
    bundle are refused first, with ``DominoDeployError``. Redeploying a model
    server prints a note that the workflow App still calls its old instance.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings.
        target: One App, or ``"all"``.
        launch_script: Absolute path of ``domino/app.sh``.
        sleep: Waits between polls; tests and dry runs pass a no-op.

    Raises:
        DominoDeployError: If the topology does not use ``target``.

    Example:
        >>> deploy(client, DeploySettings(), "all", launch_script=path)
    """
    refuse_secrets(settings)
    needed = apps_for_topology(settings.topology)
    if target != "all" and target not in needed:
        names = ", ".join(f"rag-{part}" for part in needed)
        raise DominoDeployError(
            f"the {settings.topology} topology does not use App rag-{target}; "
            f"it needs: {names}"
        )
    parts = needed if target == "all" else [target]
    refuse_placeholders(settings, parts)
    _check_deploy_settings(settings, parts)
    for part in parts:
        urls = model_server_urls(client, settings) if part == "workflow" else {}
        publish_app(
            client, settings, part, urls, launch_script=launch_script, sleep=sleep
        )
    if target in ("embedder", "reranker"):
        _note_the_workflow_calls_the_old_instance(client, target)


def _note_the_workflow_calls_the_old_instance(
    client: DominoAppsClient, part: AppPart
) -> None:
    # The workflow App holds the model server's instance URL, which a redeploy changes.
    if client.find("rag-workflow") is not None:
        print(
            f"rag-workflow still calls the previous rag-{part} instance: "
            "run make deploy-workflow"
        )


def _check_deploy_settings(settings: DeploySettings, parts: list[AppPart]) -> None:
    for part in parts:
        if not settings.for_app(part).environment_id:
            raise DominoDeployError(
                f"No environment_id for App rag-{part}: set defaults.environment_id "
                f"(or apps.{part}.environment_id) in {deploy_file()}; it is the id "
                "of the Domino compute environment the App runs in"
            )
    if settings.model_server != "ovms" or parts == ["workflow"]:
        return
    if not settings.ovms_bundle or not os.path.exists(settings.ovms_bundle):
        raise DominoDeployError(
            f"ovms_bundle={settings.ovms_bundle} does not exist: set it to "
            "the OVMS bundle folder (bin/ovms, lib/) or its .tar.gz, at a path the "
            "Apps can read"
        )


def model_server_urls(
    client: DominoAppsClient, settings: DeploySettings
) -> dict[AppPart, str]:
    """Find where the model server Apps the workflow App calls answer.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings, for the topology and Domino host.

    Returns:
        Each model server part's endpoint URL.

    Raises:
        DominoDeployError: If one of them is not deployed or not Running.

    Example:
        >>> model_server_urls(client, DeploySettings())
        {'embedder': ..., ...}
    """
    model_servers = apps_for_topology(settings.topology)[:-1]  # workflow is last
    return {
        part: endpoint_url(client, require_app(client, part), settings.domino_host)
        for part in model_servers
    }


def require_app(client: DominoAppsClient, part: AppPart) -> DominoApp:
    """Find the deployed ``rag-<part>`` App.

    Args:
        client: Apps API client for the project.
        part: The App to find.

    Returns:
        The App.

    Raises:
        DominoDeployError: If it is not deployed; the message names the make
            target.

    Example:
        >>> require_app(client, "workflow").url
        'https://domino.example/apps/a1'
    """
    app = client.find(f"rag-{part}")
    if app is None:
        raise DominoDeployError(
            f"App rag-{part} is not deployed in project {client.project_id}: "
            f"run make deploy-{part}"
        )
    return app
