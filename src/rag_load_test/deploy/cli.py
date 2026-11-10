"""rag-deploy: publish a topology's Apps on Domino 6.2 and show their status.

What to publish is decided in ``publish.py`` and with which settings in
``settings.py``; the calls to Domino's Apps API are in ``apps_client.py``. The
model server Apps are published first, then the workflow App with their URLs.

Example::

    rag-deploy deploy all      # every App the topology needs, in order
    rag-deploy status          # one line per App, then the workflow App's /readyz
    rag-deploy --dry-run deploy all   # the same, against an in-process fake Domino
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import httpx
from pydantic import ValidationError
from starlette.types import ASGIApp

from .apps_client import DominoAppsClient, DominoDeployError
from .apps_reference import AppCreationRequest
from .fake_domino import FakeDominoApps, create_fake_domino, fake_domino_http
from .publish import deploy, launch_script_in, require_app
from .settings import DeploySettings, load_app_settings
from .status import print_status

DRY_RUN_PROJECT_ID = "dry-run-project"
HTTP_TIMEOUT_S = 30.0
HTTP_RETRIES = 2


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
