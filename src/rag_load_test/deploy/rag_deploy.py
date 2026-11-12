"""rag-deploy: publish a topology's Apps on Domino 6.2 and show their status.

What to publish is decided in ``publish.py`` and with which settings in
``deploy_settings.py`` (from ``deploy.yaml``); the calls to Domino's Apps API are in
``apps_client.py``, and where a running App answers in ``endpoint_url.py``. The
model server Apps are published first, then the workflow App with their URLs.

Example:
    >>> main(["deploy", "all"])  # every App the topology needs, in order
    >>> main(["status"])  # one line per App, then the workflow App's /readyz
    >>> main(["--dry-run", "deploy", "all"])  # against an in-process fake Domino
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from typing import get_args

import httpx
from pydantic import ValidationError
from starlette.types import ASGIApp

from fake_domino.api_shapes import AppAutoscalingSpecification, AppCreationRequest
from fake_domino.server import create_fake_domino, fake_domino_http
from fake_domino.state import FakeDominoApps

from ..domain.roles import AppPart
from .app_id_cache import APP_ID_CACHE_FILE, AppIdCache
from .apps_client import DominoAppsClient, DominoDeployError
from .deploy_settings import DeploySettings, deploy_file
from .endpoint_url import endpoint_url
from .publish import CONFIGURATION_TYPES, deploy, launch_script_in, require_app
from .status import print_status

DRY_RUN_PROJECT_ID = "dry-run-project"


def main(argv: list[str] | None = None, *, fake_domino: ASGIApp | None = None) -> None:
    """Run ``rag-deploy``; expected failures end through ``parser.error`` (exit 2).

    Args:
        argv: The command-line arguments; None reads ``sys.argv``.
        fake_domino: Replaces Domino with an in-process fake (``--dry-run``
            brings its own): no ``DOMINO_API_PROXY`` needed, and waiting never
            sleeps.

    Example:
        >>> main(["--dry-run", "deploy", "all"])
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run_apps = FakeDominoApps()  # only used with --dry-run
    if args.dry_run:
        fake_domino = create_fake_domino(dry_run_apps)
    sleep = time.sleep if fake_domino is None else _no_sleep
    try:
        settings = DeploySettings()
        cache = AppIdCache(None if args.dry_run else APP_ID_CACHE_FILE)
        with _apps_client(settings, fake_domino, cache) as client:
            _run_command(args, client, settings, sleep)
    except (DominoDeployError, ValidationError) as exc:
        parser.error(str(exc))
    if args.dry_run and args.command == "deploy":
        print(dry_run_report(dry_run_apps.creates))


def _no_sleep(_seconds: float) -> None:
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
    parts = list(get_args(AppPart))  # embedder, reranker, workflow
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
        deploy(client, settings, args.target, launch_script=launch_script, sleep=sleep)
    if args.command == "status":
        print_status(client, settings)
    if args.command == "url":
        app = require_app(client, args.part)
        print(endpoint_url(client, app, settings.domino_host))


def dry_run_report(creates: list[AppCreationRequest]) -> str:
    """Describe what a dry run would have published.

    Args:
        creates: The create requests the fake Domino received, in order.

    Returns:
        Each App's kind, environment, tier, autoscaling and entry script, as text.

    Example:
        >>> print(dry_run_report(fake_apps.creates))
        Dry run: nothing was sent ...
    """
    kinds = {ctype: kind for kind, ctype in CONFIGURATION_TYPES.items()}
    lines = ["", "Dry run: nothing was sent to Domino. rag-deploy would publish:"]
    for create in creates:
        version, ctype = create.version, create.configurationType or "STANDARD"
        lines += [
            "",
            create.name,
            f"  kind: {kinds[ctype]} (configurationType {ctype})",
            f"  environmentId: {version.environmentId}",
            f"  hardwareTierId: {version.hardwareTierId or '(the project default)'}",
            f"  autoscaling: {_autoscaling_text(version.autoscalingSpecification)}",
            "  entryScript:",
        ]
        lines += [f"    {line}" for line in (version.entryScript or "").splitlines()]
    return "\n".join(lines)


def _autoscaling_text(spec: AppAutoscalingSpecification | None) -> str:
    if spec is None or not spec.enabled:
        return "off"
    fields = spec.model_dump(exclude_none=True, exclude={"enabled"})
    return ", ".join(f"{name} {value}" for name, value in fields.items())


def _apps_client(
    settings: DeploySettings, fake_domino: ASGIApp | None, cache: AppIdCache
) -> DominoAppsClient:
    if fake_domino is not None:
        project_id = settings.project_id or DRY_RUN_PROJECT_ID
        return DominoAppsClient(fake_domino_http(fake_domino), project_id, cache)
    proxy = os.environ.get("DOMINO_API_PROXY", "")
    if not proxy:
        raise DominoDeployError(_outside_workspace("DOMINO_API_PROXY"))
    if not settings.project_id:
        raise DominoDeployError(
            _outside_workspace(
                "DOMINO_PROJECT_ID", f" and project_id is empty in {deploy_file()}"
            )
        )
    http = httpx.Client(
        base_url=proxy,
        timeout=settings.http_timeout_s,
        transport=httpx.HTTPTransport(retries=settings.http_retries),
    )
    return DominoAppsClient(http, settings.project_id, cache)


def _outside_workspace(variable: str, also: str = "") -> str:
    return (
        f"{variable} is not set{also}: rag-deploy runs in a Domino Workspace of the "
        "project that hosts the Apps, where Domino sets it"
    )
