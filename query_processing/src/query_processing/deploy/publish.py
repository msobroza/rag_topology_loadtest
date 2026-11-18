"""Publishing a topology's Apps: their entry scripts, versions and order.

Which Apps follow from the topology alias deployed (``topology_alias.py``): the
model server Apps are published first, then the workflow App with the URLs
their running instances answer on and each component's backend. Every entry
script runs the toolkit's ``domino/app.sh`` in one of two Roles: the workflow
app, or a model server told its component and its model-server kind, and
names the use case's directory, where the role's command finds the use case's
files. What else a kind's launch script reads, and what a kind refuses, is the
kind's class's to say (``model_server_kinds/``), so the flow here is the same
for every kind. Every entry script also exports the overrides, this site's
values of the workflow file (``overrides.py``).
"""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

from ..domain.roles import MODEL_SERVER_ROLE, WORKFLOW_APP, WORKFLOW_ROLE, AppPart
from .apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    wait_until_running,
)
from .deploy_refusals import refuse_placeholders, refuse_secrets
from .deploy_settings import (
    SERVED,
    AppKind,
    AppSettings,
    Autoscaling,
    DeploySettings,
    deploy_file,
)
from .endpoint_url import endpoint_url
from .model_server_kinds.base_model_server_kind import model_server_kind
from .overrides import (
    check_overrides,
    override_exports,
    refuse_workflow_settings_in_free_lists,
)
from .topology_alias import ResolvedTopology, resolve_topology
from .workflow_components import component_variable

# The toolkit's directory in the repository, a use case's sibling, and its
# launch script within it.
TOOLKIT_DIRECTORY = "query_processing"
LAUNCH_SCRIPT = "domino/app.sh"
# What tells app.sh where to run the role's command: one of qp-deploy's own
# exports, which win over the settings and the overrides.
USE_CASE_DIR_VARIABLE = "QP_USE_CASE_DIR"
# An Agent is an App whose configurationType is AISYSTEM: the only API-level
# difference (assumed to be what the UI's "Deploy Agent" sends).
CONFIGURATION_TYPES: dict[AppKind, str] = {"app": "STANDARD", "agent": "AISYSTEM"}
# Exported by an Agent unless its overrides or its settings set them: Domino
# tracing (domino.tracing of the workflow file), and the [domino] extra that
# brings its library (a variable of the launch script).
AGENT_EXPORTS = {"QP_DOMINO__TRACING": "1", "QP_BOOTSTRAP_EXTRAS": "domino"}


@dataclass(frozen=True)
class LaunchPaths:
    """Where every App starts: the toolkit's launch script and the use case.

    Both are absolute: the Apps mount the repository at the same path as the
    Workspace (for example /repos/<project>) but do not start in it.

    Example:
        >>> LaunchPaths("/repos/p/query_processing/domino/app.sh", "/repos/p/uc")
    """

    script: str  # the toolkit's domino/app.sh
    use_case_dir: str  # where app.sh runs the role's command: qp-deploy's own


def launch_paths(use_case_dir: str) -> LaunchPaths:
    """Find the toolkit's launch script, going up from the use case's directory.

    qp-deploy runs from a use case's directory, a sibling of the toolkit's in
    the repository, so the launch script is the first
    ``query_processing/domino/app.sh`` found from there up.

    Args:
        use_case_dir: The use case's directory, absolute: the working directory.

    Returns:
        The launch script found, and ``use_case_dir``.

    Raises:
        DominoDeployError: If no directory from ``use_case_dir`` up holds the
            launch script; the message names where it looked.

    Example:
        >>> launch_paths("/repos/rag/my_use_case").script
        '/repos/rag/query_processing/domino/app.sh'
    """
    start = Path(use_case_dir)
    wanted = Path(TOOLKIT_DIRECTORY, LAUNCH_SCRIPT)
    for directory in (start, *start.parents):
        if (directory / wanted).is_file():
            return LaunchPaths(str(directory / wanted), use_case_dir)
    raise DominoDeployError(
        f"no {wanted} in {use_case_dir} or any directory above it: run qp-deploy "
        "from a use case's directory in the repository (for example "
        f"/repos/<project>/<use case>, beside {TOOLKIT_DIRECTORY}/), where the "
        "Apps run the role's command"
    )


def entry_script(
    app: AppSettings,
    settings: DeploySettings,
    topology: ResolvedTopology,
    urls: Mapping[AppPart, str],
    launch: LaunchPaths,
) -> str:
    r"""Write the shell that exports one App's settings, then runs the launch script.

    Each App carries its own environment variables: ``app.settings`` first, then
    every override, then the computed ones, which therefore win, the use case's
    directory last among them.

    Args:
        app: The App's merged settings.
        settings: The whole deploy settings, for the overrides and what a
            model-server kind reads.
        topology: The topology alias deployed, resolved.
        urls: Where the model server Apps answer (workflow App only).
        launch: The launch script to run, and the use case's directory.

    Returns:
        The entry script, one ``export`` per line, then ``exec bash``.

    Example:
        >>> app = settings.for_app("reranker")
        >>> entry_script(app, settings, topology, {}, launch)
        'export QP_ROLE=model-server\n...\nexec bash /repos/rag/query_processing/...'
    """
    overrides = override_exports(settings)
    given = {**app.settings, **overrides}  # what the computed defaults yield to
    if app.part == WORKFLOW_APP:
        computed = _workflow_exports(topology, urls, given)
    else:
        computed = _model_server_exports(app.part, settings, topology)
    if app.kind == "agent":
        computed |= {k: v for k, v in AGENT_EXPORTS.items() if k not in given}
    # The generic environment has nothing installed: app.sh installs it first,
    # the toolkit and the use case, then runs the role's command in the latter.
    exports = [
        *app.settings.items(),
        *overrides.items(),
        *computed.items(),
        (USE_CASE_DIR_VARIABLE, launch.use_case_dir),
        ("QP_BOOTSTRAP", "1"),
    ]
    lines = [f"export {name}={shlex.quote(value)}" for name, value in exports]
    return "\n".join([*lines, f"exec bash {shlex.quote(launch.script)}"])


def _model_server_exports(
    part: AppPart, settings: DeploySettings, topology: ResolvedTopology
) -> dict[str, str]:
    # app.sh starts the kind's own launch script, which serves the component.
    kind = topology.kinds[part]
    exports = {
        "QP_ROLE": MODEL_SERVER_ROLE,
        "QP_MODEL_SERVER_COMPONENT": part,
        "QP_MODEL_SERVER_KIND": kind,
    }
    return exports | model_server_kind(kind).launch_exports(settings)


def _workflow_exports(
    topology: ResolvedTopology,
    urls: Mapping[AppPart, str],
    given: Mapping[str, str],
) -> dict[str, str]:
    # The backends below place each component (ADR 0003); the alias is only the
    # label the workflow app reports as its X-RAG-Topology header. The run
    # label is a default: a run_label under overrides names a variant.
    exports: dict[str, str] = {
        "QP_ROLE": WORKFLOW_ROLE,
        "QP_TOPOLOGY_ALIAS": topology.alias,
    }
    for name, placement in topology.placements.items():
        exports[component_variable(name, "backend")] = placement
        if placement == SERVED:
            # The settings take the full route; an App URL may end with a "/".
            url = urls[name].rstrip("/") + topology.served_route(name)
            exports[component_variable(name, "served", "url")] = url
    if "QP_RUN_LABEL" not in given:
        exports["QP_RUN_LABEL"] = topology.run_label()
    return exports


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
    launch: LaunchPaths,
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
        launch: The launch script the App runs, and the use case's directory.
        sleep: Waits between polls; tests and dry runs pass a no-op.

    Returns:
        The App, Running.

    Raises:
        DominoDeployError: If the existing App is of another kind (App or Agent),
            a model server App the workflow App calls is not Running, or the App
            does not come up.

    Example:
        >>> publish_app(client, settings, topology, "reranker", launch=launch)
    """
    app_settings = settings.for_app(part)
    urls: dict[AppPart, str] = {}
    if part == WORKFLOW_APP:
        urls = model_server_urls(client, settings, topology)
    script = entry_script(app_settings, settings, topology, urls, launch)
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
    launch: LaunchPaths,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Publish ``target``, or every App the topology alias needs, in order.

    Settings holding secrets, a value of the workflow file in a free list,
    overrides the workflow settings refuse (``check_overrides``), a topology
    alias that does not resolve against the workflow file
    (``resolve_topology``), placeholders, no environment id, a default
    environment revision without the default environment, or settings a
    model-server kind of the Apps cannot start with (a missing OVMS bundle) are
    refused first, with ``DominoDeployError``. Redeploying a model server
    prints a note that the workflow App still calls its old instance.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings.
        target: One App, or ``"all"``.
        launch: The launch script every App runs, and the use case's directory.
        sleep: Waits between polls; tests and dry runs pass a no-op.

    Raises:
        DominoDeployError: If the topology does not use ``target``.

    Example:
        >>> deploy(client, DeploySettings(), "all", launch=launch_paths(os.getcwd()))
    """
    refuse_secrets(settings)
    refuse_workflow_settings_in_free_lists(settings)
    check_overrides(settings)
    topology = resolve_topology(settings)
    needed = topology.app_parts()
    if target != "all" and target not in needed:
        _refuse_an_app_the_topology_does_not_use(settings, target, needed)
    parts = needed if target == "all" else [target]
    refuse_placeholders(settings, parts)
    _check_deploy_settings(settings, topology, parts)
    for part in parts:
        publish_app(client, settings, topology, part, launch=launch, sleep=sleep)
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


def _check_deploy_settings(
    settings: DeploySettings, topology: ResolvedTopology, parts: list[AppPart]
) -> None:
    _refuse_a_default_revision_alone(settings)
    for app in map(settings.for_app, parts):
        if not app.environment_id:
            raise DominoDeployError(
                f"No environment_id for App {app.name}: set defaults.environment_id "
                f"(or apps.{app.part}.environment_id) in {deploy_file()}; it is the "
                "id of the Domino compute environment the App runs in"
            )
    for part in parts:
        if part in topology.kinds:  # each model server App's kind checks its own
            model_server_kind(topology.kinds[part]).refuse_unusable_settings(settings)


def _refuse_a_default_revision_alone(settings: DeploySettings) -> None:
    # It goes only with the default environment, so without one it would be
    # silently dropped. An App's own revision without any environment is caught
    # by the environment_id check, since that App has none.
    defaults = settings.defaults
    if not defaults.sets_the_revision() or defaults.environment_id:
        return
    key, revision = (
        ("environment_revision", defaults.environment_revision)
        if defaults.environment_revision
        else ("environment_revision_id", defaults.environment_revision_id)
    )
    raise DominoDeployError(
        f"defaults.{key} {revision!r} in {deploy_file()} has no "
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
