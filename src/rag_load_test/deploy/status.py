"""``rag-deploy status``: one line per App, then the workflow App's /readyz.

A Running App shows where its instance answers; any other, its page in Domino.
"""

from __future__ import annotations

from http import HTTPStatus

import httpx

from ..domain.errors import describe_error
from .apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    json_list,
    json_mapping,
)
from .endpoint_url import endpoint_url
from .publish import apps_for_topology
from .settings import DeploySettings


def print_status(client: DominoAppsClient, settings: DeploySettings) -> None:
    """Print one line per App the topology needs, then the workflow App's /readyz.

    The /readyz lines appear only when the workflow App is Running.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings (topology, Domino host, run token URL).

    Example:
        >>> print_status(client, DeploySettings())
        rag-workflow   Running   https://...
    """
    endpoint: str | None = None
    for part in apps_for_topology(settings.topology):  # the workflow App is last
        app = client.find(f"rag-{part}")
        endpoint = None
        if app is not None and app.status == "Running":
            endpoint = endpoint_url(client, app, settings.domino_host)
        print(_status_line(f"rag-{part}", app, endpoint))
    if endpoint is not None:
        lines = readiness_lines(client.http, endpoint, settings.run_token_url)
        print("\n".join(lines))


def _status_line(name: str, app: DominoApp | None, endpoint: str | None) -> str:
    if app is None:
        return f"{name:<14} not deployed"
    return f"{name:<14} {app.status or 'no instance':<14} {endpoint or app.url}"


def readiness_lines(http: httpx.Client, url: str, token_url: str) -> list[str]:
    """Report the workflow App's /readyz checks and run headers.

    Never raises: the App table is the main output of ``rag-deploy status``.

    Args:
        http: Client the token and /readyz requests go through.
        url: Where the workflow App's instance answers.
        token_url: The run's token service, whose token is sent as a Bearer.

    Returns:
        The status and run headers, then one line per check; or one line saying
        why /readyz was not checked.

    Example:
        >>> readiness_lines(http, endpoint, settings.run_token_url)
        ['readyz: ...', ...]
    """
    try:
        headers = {"Authorization": f"Bearer {_run_token(http, token_url)}"}
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


def _run_token(http: httpx.Client, token_url: str) -> str:
    response = http.get(token_url)
    token = response.text.strip()
    if response.status_code != HTTPStatus.OK or not token:
        raise DominoDeployError(
            f"run token from {token_url}: HTTP {response.status_code}"
        )
    return token
