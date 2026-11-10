"""rag-deploy: publish a topology's Apps on Domino 6.2 and show their status.

What to publish and with which settings is decided here; the calls to Domino's
Apps API are in ``domino_apps.py``. The model server Apps are published first,
then the workflow App with their URLs.

Example::

    rag-deploy deploy all      # every App the topology needs, in order
    rag-deploy status          # one line per App, then the workflow App's /readyz
    rag-deploy --dry-run deploy all   # the same, against an in-process fake Domino
"""

from __future__ import annotations

import argparse
import os
import shlex
import time
from collections.abc import Callable, Mapping
from typing import Literal

import httpx
from dotenv import dotenv_values
from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import ASGIApp

from .domino_apps import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    json_list,
    json_mapping,
    wait_until_running,
)
from .domino_apps_reference import AppCreationRequest
from .fake_domino import FakeDominoApps, create_fake_domino, fake_domino_http
from .models import describe_error
from .ovms import DOMINO_TOKEN_URL
from .topology import Topology

AppPart = Literal["embedder", "reranker", "workflow"]
ModelServer = Literal["fastapi", "ovms"]
LAUNCH_SCRIPT = "domino/app.sh"  # relative to the repository root
DRY_RUN_PROJECT_ID = "dry-run-project"
DEPLOY_ENV_FILE = "deploy.env"
HTTP_TIMEOUT_S = 30.0
HTTP_RETRIES = 2

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


def print_status(client: DominoAppsClient, settings: DeploySettings) -> None:
    """One line per App the topology needs, then the running workflow App's /readyz.

    Example::

        print_status(client, DeploySettings())   # "rag-workflow   Running   https://..."
    """
    app: DominoApp | None = None
    for part in apps_for_topology(settings.topology):  # the workflow App is last
        app = client.find(f"rag-{part}")
        print(_status_line(f"rag-{part}", app))
    if app is not None and app.status == "Running":
        print("\n".join(readiness_lines(client.http, app.url)))


def _status_line(name: str, app: DominoApp | None) -> str:
    if app is None:
        return f"{name:<14} not deployed"
    return f"{name:<14} {app.status or 'no instance':<14} {app.url}"


def readiness_lines(http: httpx.Client, url: str) -> list[str]:
    """The workflow App's /readyz checks and run headers, or one line saying why not.

    Never raises: the App table is the main output of ``rag-deploy status``.

    Example::

        readiness_lines(http, "https://domino.example/apps/a1")   # ["readyz: ...", ...]
    """
    try:
        headers = {"Authorization": f"Bearer {_run_token(http)}"}
        response = http.get(f"{url}/readyz", headers=headers)
        checks = json_list(json_mapping(response.json()).get("checks"))
    except DominoDeployError as exc:
        return [f"readyz: not checked ({exc})"]
    except (httpx.HTTPError, ValueError) as exc:
        return [f"readyz: not checked ({describe_error(exc)})"]
    lines = [
        f"readyz: HTTP {response.status_code}, "
        f"X-RAG-Topology: {response.headers.get('X-RAG-Topology')}, "
        f"X-RAG-Run-Label: {response.headers.get('X-RAG-Run-Label')}"
    ]
    for check in map(json_mapping, checks):
        verdict = "ok" if check.get("ok") is True else "FAIL"
        lines.append(f"  {check.get('name')}: {verdict} {check.get('detail')}")
    return lines


def _run_token(http: httpx.Client) -> str:
    response = http.get(DOMINO_TOKEN_URL)
    token = response.text.strip()
    if response.status_code != 200 or not token:
        raise DominoDeployError(
            f"run token from {DOMINO_TOKEN_URL}: HTTP {response.status_code}"
        )
    return token


def main(argv: list[str] | None = None, *, fake_domino: ASGIApp | None = None) -> None:
    """``rag-deploy``: expected failures end through ``parser.error`` (exit 2).

    ``fake_domino`` (or ``--dry-run``, which brings its own) replaces Domino with
    an in-process fake: no ``DOMINO_API_PROXY`` needed, and waiting never sleeps.

    Example::

        main(["--dry-run", "deploy", "all"])
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run_apps = FakeDominoApps()  # only used with --dry-run
    if args.dry_run:
        fake_domino = create_fake_domino(dry_run_apps)
    sleep = time.sleep if fake_domino is None else _no_sleep
    try:
        settings = DeploySettings()
        with _apps_client(settings, fake_domino) as client:
            _run_command(args, client, settings, sleep)
    except (DominoDeployError, ValidationError) as exc:
        parser.error(str(exc))
    if args.dry_run and args.command == "deploy":
        print(dry_run_report(dry_run_apps.creates))


def _no_sleep(seconds: float) -> None:
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-deploy", description="Publish this project's Apps on Domino 6.2."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run against an in-process fake Domino: nothing is sent to Domino",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    parts = ["embedder", "reranker", "workflow"]
    commands.add_parser("deploy").add_argument("target", choices=["all", *parts])
    commands.add_parser("status")
    commands.add_parser("url").add_argument("part", choices=parts)
    return parser


def _run_command(
    args: argparse.Namespace,
    client: DominoAppsClient,
    settings: DeploySettings,
    sleep: Callable[[float], None],
) -> None:
    if args.command == "deploy":
        launch_script = launch_script_in(os.getcwd())
        app_settings = load_app_settings()
        deploy(
            client,
            settings,
            args.target,
            app_settings,
            launch_script=launch_script,
            sleep=sleep,
        )
    if args.command == "status":
        print_status(client, settings)
    if args.command == "url":
        print(require_app(client, args.part).url)


def dry_run_report(creates: list[AppCreationRequest]) -> str:
    """What a dry run would have published: each App's settings and entry script.

    Example::

        print(dry_run_report(fake_apps.creates))   # "Dry run: nothing was sent ..."
    """
    lines = ["", "Dry run: nothing was sent to Domino. rag-deploy would publish:"]
    for create in creates:
        version = create.version
        lines += ["", create.name, f"  environmentId: {version.environmentId}"]
        if version.hardwareTierId:
            lines.append(f"  hardwareTierId: {version.hardwareTierId}")
        lines.append("  entryScript:")
        lines += [f"    {line}" for line in (version.entryScript or "").splitlines()]
    return "\n".join(lines)


def _apps_client(
    settings: DeploySettings, fake_domino: ASGIApp | None
) -> DominoAppsClient:
    if fake_domino is not None:
        project_id = settings.project_id or DRY_RUN_PROJECT_ID
        return DominoAppsClient(fake_domino_http(fake_domino), project_id)
    proxy = os.environ.get("DOMINO_API_PROXY", "")
    if not proxy:
        raise DominoDeployError(_outside_workspace("DOMINO_API_PROXY"))
    if not settings.project_id:
        raise DominoDeployError(
            _outside_workspace(
                "DOMINO_PROJECT_ID", " and neither is RAG_DEPLOY_PROJECT_ID"
            )
        )
    http = httpx.Client(
        base_url=proxy,
        timeout=HTTP_TIMEOUT_S,
        transport=httpx.HTTPTransport(retries=HTTP_RETRIES),
    )
    return DominoAppsClient(http, settings.project_id)


def _outside_workspace(variable: str, also: str = "") -> str:
    return (
        f"{variable} is not set{also}: rag-deploy runs in a Domino Workspace of the "
        "project that hosts the Apps, where Domino sets it"
    )
