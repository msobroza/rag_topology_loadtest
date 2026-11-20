"""``qp-deploy status``: one line per App, then the workflow App's /readyz.

A Running App shows where its instance answers; any other, its page in Domino.
The /readyz request carries the run token, asked for through the ``domino`` key
source the composition opens (``run_token_source``), under the deploy tool's own
timeout and retries: the one module of ``deploy`` that imports it for that, and
the key source's port to name its type (README rule 1).
"""

from __future__ import annotations

import httpx

from ..adapters.key_source.base_key_source import BaseKeySource
from ..composition.standalone_key_source import run_token_source
from ..domain.errors import RagDependencyError, describe_error
from .apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
    json_list,
    json_mapping,
)
from .deploy_settings import DeploySettings
from .endpoint_url import endpoint_url
from .topology_alias import resolve_topology


def print_status(client: DominoAppsClient, settings: DeploySettings) -> None:
    """Print one line per App the topology needs, then the workflow App's /readyz.

    Each App is looked up by its name, ``<app_name_prefix>-<part>``. The /readyz
    lines appear only when the workflow App is Running.

    Args:
        client: Apps API client for the project.
        settings: The deploy settings (topology alias, Domino host, run token
            URL, the timeout and retries of a call).

    Example:
        >>> print_status(client, DeploySettings())
        rag-workflow   Running   https://...
    """
    endpoint: str | None = None
    parts = resolve_topology(settings).app_parts()
    names = [settings.app_name(part) for part in parts]
    width = max(map(len, names)) + 2  # a column as wide as the longest name
    for name in names:  # the workflow App is last
        app = client.find(name)
        endpoint = None
        if app is not None and app.status == "Running":
            endpoint = endpoint_url(client, app, settings.domino_host)
        print(_status_line(f"{name:<{width}}", app, endpoint))
    if endpoint is not None:
        source = run_token_source(
            client.http,
            token_url=settings.run_token_url,
            timeout_s=settings.http_timeout_s,
            retries=settings.http_retries,
        )
        print("\n".join(readiness_lines(client.http, endpoint, source)))


def _status_line(column: str, app: DominoApp | None, endpoint: str | None) -> str:
    if app is None:
        return f"{column} not deployed"
    return f"{column} {app.status or 'no instance':<14} {endpoint or app.url}"


def readiness_lines(http: httpx.Client, url: str, source: BaseKeySource) -> list[str]:
    """Report the workflow App's /readyz checks and run headers.

    Never raises: the App table is the main output of ``qp-deploy status``.

    Args:
        http: Client the /readyz request goes through.
        url: Where the workflow App's instance answers.
        source: The ``domino`` key source, whose run token is sent as a Bearer.

    Returns:
        The status and run headers, then one line per check; or one line saying
        why /readyz was not checked.

    Example:
        >>> readiness_lines(http, endpoint, source)
        ['readyz: ...', ...]
    """
    try:
        headers = {"Authorization": f"Bearer {source.key(may_ask=True)}"}
        response = http.get(f"{url}/readyz", headers=headers)
        checks = json_list(json_mapping(response.json()).get("checks"))
    except RagDependencyError as exc:  # the token URL never gave a token
        return [f"readyz: not checked (run token from {exc.target}: {exc.detail})"]
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
