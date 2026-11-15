"""Publishing a topology's Apps: their entry scripts, versions and order.

Which Apps follow from the topology alias deployed (``topology_alias.py``): the
model server Apps are published first, then the workflow App with the URLs
their running instances answer on and each component's backend.
"""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Callable, Mapping
from typing import Literal, NoReturn

from ..domain.roles import (
    MODEL_SERVER_ROUTES,
    WORKFLOW_APP,
    AppPart,
    ModelServerKind,
    model_server_role,
    workflow_role,
)
from ..domain.topology import SERVED
from .apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    wait_until_running,
)
from .deploy_settings import (
    AppKind,
    AppSettings,
    Autoscaling,
    DeploySettings,
    deploy_file,
    refuse_placeholders,
    refuse_secrets,
)
from .endpoint_url import endpoint_url
from .topology_alias import ResolvedTopology, resolve_topology
from .workflow_components import component_variable

LAUNCH_SCRIPT = "domino/app.sh"  # relative to the repository root
# An Agent is an App whose configurationType is AISYSTEM: the only API-level
# difference (assumed to be what the UI's "Deploy Agent" sends).
CONFIGURATION_TYPES: dict[AppKind, str] = {"app": "STANDARD", "agent": "AISYSTEM"}
# Exported by an Agent unless its settings set them: Domino tracing, and the
# [domino] extra that brings its library.
AGENT_EXPORTS = {"RAG_DOMINO_TRACING": "1", "RAG_BOOTSTRAP_EXTRAS": "domino"}


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
    topology: ResolvedTopology,
    urls: Mapping[AppPart, str],
    launch_script: str,
) -> str:
    r"""Write the shell that exports one App's settings, then runs ``launch_script``.

    Each App carries its own environment variables: ``app.settings`` first, then
    the computed ones, which therefore win.

    Args:
        app: The App's merged settings.
        settings: The whole deploy settings (model server, bundle).
        topology: The topology alias deployed, resolved.
        urls: Where the model server Apps answer (workflow App only).
        launch_script: Absolute path of ``domino/app.sh``.

    Returns:
        The entry script, one ``export`` per line, then ``exec bash``.

    Example:
        >>> app = settings.for_app("reranker")
        >>> entry_script(app, settings, topology, {}, "/repos/rag/domino/app.sh")
        'export RAG_ROLE=fastapi-reranker\n...\nexec bash /repos/rag/domino/app.sh'
    """
    if app.part == WORKFLOW_APP:
        computed = _workflow_exports(settings, topology, urls, app.settings)
    else:
        computed = _model_server_exports(app.part, settings)
    if app.kind == "agent":
        computed |= {k: v for k, v in AGENT_EXPORTS.items() if k not in app.settings}
    # The generic environment has nothing installed: app.sh installs it first.
    exports = [*app.settings.items(), *computed.items(), ("RAG_BOOTSTRAP", "1")]
    lines = [f"export {name}={shlex.quote(value)}" for name, value in exports]
    return "\n".join([*lines, f"exec bash {shlex.quote(launch_script)}"])


def _model_server_exports(part: AppPart, settings: DeploySettings) -> dict[str, str]:
    role = model_server_role(settings.model_server, part)
    exports: dict[str, str] = {"RAG_ROLE": role}
    if settings.model_server == "ovms":
        exports["RAG_OVMS_BUNDLE"] = settings.ovms_bundle
    return exports


def _workflow_exports(
    settings: DeploySettings,
    topology: ResolvedTopology,
    urls: Mapping[AppPart, str],
    app_settings: Mapping[str, str],
) -> dict[str, str]:
    # The workflow app still applies its own topology rule (ADR 0003's first
    # step): it is told a topology whose rule moves none of the backends below,
    # and app.sh pairs the Role with it.
    told = topology.topology_variable()
    exports: dict[str, str] = {"RAG_ROLE": workflow_role(told), "RAG_TOPOLOGY": told}
    for name, placement in topology.placements.items():
        exports[component_variable(name, "backend")] = placement
        if placement == SERVED:
            exports[component_variable(name, "served", "url")] = _served_url(
                urls[name], topology.components[name].type, settings.model_server
            )
    if "RAG_RUN_LABEL" not in app_settings:
        exports["RAG_RUN_LABEL"] = topology.run_label(settings.model_server)
    return exports


def _served_url(app_url: str, component_type: str, kind: ModelServerKind) -> str:
    # The settings take the full route; an App URL may end with a "/". The type
    # is one the kind serves: resolve_topology refused any other.
    return app_url.rstrip("/") + MODEL_SERVER_ROUTES[kind][component_type]


def version_body(app: AppSettings, script: str) -> dict[str, object]:
    """Build the version object: script, environment, and what else is set.

    The environment revision, the hardware tier and the autoscaling are sent
    only when the App's settings set them.

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
    # The field's name in Domino's Public API spec (6.3.1), optional there: a
    # version sent without it runs the revision Domino chooses.
    if app.environment_revision_id:
        body["environmentRevisionId"] = app.environment_revision_id
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
    topology: ResolvedTopology,
    part: AppPart,
    *,
    launch_script: str,
    sleep: Callable[[float], None] = time.sleep,
) -> DominoApp:
    """Create ``part``'s App or post a new version to it, then wait until Running.

    The App is found and created by its name, ``<app_name_prefix>-<part>``. The
    workflow App is handed where the model server Apps of the topology answer.

    Prints the App's endpoint once Running.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings.
        topology: The topology alias deployed, resolved.
        part: The App to publish.
        launch_script: Absolute path of ``domino/app.sh``.
        sleep: Waits between polls; tests and dry runs pass a no-op.

    Returns:
        The App, Running.

    Raises:
        DominoDeployError: If the existing App is of another kind (App or Agent),
            a model server App the workflow App calls is not Running, or the App
            does not come up.

    Example:
        >>> publish_app(client, settings, topology, "reranker", launch_script=path)
    """
    app_settings = settings.for_app(part)
    urls: dict[AppPart, str] = {}
    if part == WORKFLOW_APP:
        urls = model_server_urls(client, settings, topology)
    script = entry_script(app_settings, settings, topology, urls, launch_script)
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
    """Publish ``target``, or every App the topology alias needs, in order.

    Settings holding secrets, a topology alias that does not resolve against
    the workflow file (``resolve_topology``), placeholders, no environment id, a
    default environment revision without the default environment, or a missing
    OVMS bundle are refused first, with ``DominoDeployError``. Redeploying a
    model server prints a note that the workflow App still calls its old
    instance.

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
    topology = resolve_topology(settings)
    needed = topology.app_parts()
    if target != "all" and target not in needed:
        _refuse_an_app_the_topology_does_not_use(settings, target, needed)
    parts = needed if target == "all" else [target]
    refuse_placeholders(settings, parts)
    _check_deploy_settings(settings, parts)
    for part in parts:
        publish_app(
            client, settings, topology, part, launch_script=launch_script, sleep=sleep
        )
    if target in topology.served:
        _note_the_workflow_calls_the_old_instance(client, settings, target)


def _refuse_an_app_the_topology_does_not_use(
    settings: DeploySettings, target: AppPart, needed: list[AppPart]
) -> NoReturn:
    names = ", ".join(settings.app_name(part) for part in needed)
    raise DominoDeployError(
        f"the {settings.topology} topology does not use App "
        f"{settings.app_name(target)}; it needs: {names}"
    )


def _note_the_workflow_calls_the_old_instance(
    client: DominoAppsClient, settings: DeploySettings, part: AppPart
) -> None:
    # The workflow App holds the model server's instance URL, which a redeploy changes.
    workflow = settings.app_name(WORKFLOW_APP)
    if client.find(workflow) is not None:
        print(
            f"{workflow} still calls the previous {settings.app_name(part)} "
            "instance: run make deploy-workflow"
        )


def _check_deploy_settings(settings: DeploySettings, parts: list[AppPart]) -> None:
    _refuse_a_default_revision_alone(settings)
    for app in map(settings.for_app, parts):
        if not app.environment_id:
            raise DominoDeployError(
                f"No environment_id for App {app.name}: set defaults.environment_id "
                f"(or apps.{app.part}.environment_id) in {deploy_file()}; it is the "
                "id of the Domino compute environment the App runs in"
            )
    if settings.model_server != "ovms" or parts == [WORKFLOW_APP]:
        return
    if not settings.ovms_bundle or not os.path.exists(settings.ovms_bundle):
        raise DominoDeployError(
            f"ovms_bundle={settings.ovms_bundle} does not exist: set it to "
            "the OVMS bundle folder (bin/ovms, lib/) or its .tar.gz, at a path the "
            "Apps can read"
        )


def _refuse_a_default_revision_alone(settings: DeploySettings) -> None:
    # It goes only with the default environment, so without one it would be
    # silently dropped. An App's own revision without any environment is caught
    # by the environment_id check, since that App has none.
    defaults = settings.defaults
    revision_id = defaults.environment_revision_id
    if not revision_id or defaults.environment_id:
        return
    raise DominoDeployError(
        f"defaults.environment_revision_id {revision_id!r} in {deploy_file()} has no "
        "defaults.environment_id beside it: a default revision applies only together "
        "with the default environment, so set defaults.environment_id to the "
        "environment it is a build of, or set the revision in the block of each App "
        "that sets its own environment_id"
    )


def model_server_urls(
    client: DominoAppsClient, settings: DeploySettings, topology: ResolvedTopology
) -> dict[AppPart, str]:
    """Find where the model server Apps the workflow App calls answer.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings, for the App names and Domino host.
        topology: The topology alias deployed, for its served components.

    Returns:
        Each served component's model server App URL, by component.

    Raises:
        DominoDeployError: If one of them is not deployed or not Running.

    Example:
        >>> model_server_urls(client, DeploySettings(), topology)
        {'embedder': ..., ...}
    """
    return {
        part: endpoint_url(
            client, require_app(client, settings, part), settings.domino_host
        )
        for part in topology.served
    }


def require_app(
    client: DominoAppsClient, settings: DeploySettings, part: AppPart
) -> DominoApp:
    """Find ``part``'s deployed App, by its name ``<app_name_prefix>-<part>``.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings, for the App's name.
        part: The App to find.

    Returns:
        The App.

    Raises:
        DominoDeployError: If it is not deployed; the message names the make
            target.

    Example:
        >>> require_app(client, DeploySettings(), "workflow").url
        'https://domino.example/apps/a1'
    """
    name = settings.app_name(part)
    app = client.find(name)
    if app is None:
        # The make target is named after the part, whatever the prefix.
        raise DominoDeployError(
            f"App {name} is not deployed in project {client.project_id}: "
            f"run make deploy-{part}"
        )
    return app
