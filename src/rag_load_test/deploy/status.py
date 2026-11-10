"""``rag-deploy status``: one line per App, then the workflow App's /readyz."""

from __future__ import annotations

import httpx

from ..adapters.ovms import DOMINO_TOKEN_URL
from ..domain.errors import describe_error
from .apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    json_list,
    json_mapping,
)
from .publish import apps_for_topology
from .settings import DeploySettings


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
