"""Where an App answers: ``{host}/{owner}/{project}/r/notebookSession/{instance id}``.

This path is not in the Domino 6.2 API reference: it is how the project owner
calls Apps on their Domino. The instance id changes at every redeploy (scaling
adds replicas inside the same instance, so it does not change the id).

Example:
    >>> endpoint_url(client, client.find("rag-reranker"), "")
    'https://domino.example/alice/rag/r/notebookSession/6f1c...'
"""

from __future__ import annotations

from urllib.parse import urlsplit

from .apps_client import AppInstance, DominoApp, DominoAppsClient, DominoDeployError


def endpoint_url(client: DominoAppsClient, app: DominoApp, domino_host: str) -> str:
    """Return the URL ``app``'s Running instance answers on.

    Args:
        client: Apps API client for the App's project.
        app: The App whose instance is called.
        domino_host: Scheme and host of Domino; "" takes them from the App's own
            url.

    Returns:
        ``{host}/{owner}/{project}/r/notebookSession/{instance id}``.

    Raises:
        DominoDeployError: If no host can be worked out, the App lacks its owner
            or project name, or it has no Running instance of its current
            version.

    Example:
        >>> endpoint_url(client, app, "https://domino.example")
        'https://domino.example/alice/rag/r/notebookSession/6f1c'
    """
    host = _host(app, domino_host)
    _refuse_missing_owner_or_project(app)
    instance = _running_instance(client, app)
    return f"{host}/{app.owner}/{app.project}/r/notebookSession/{instance.id}"


def _host(app: DominoApp, domino_host: str) -> str:
    if domino_host:
        return domino_host.rstrip("/")
    parts = urlsplit(app.url)
    if not (parts.scheme and parts.netloc):
        raise DominoDeployError(
            f"App {app.name} has no url to take the Domino host from: set "
            "domino_host in deploy.yaml (for example https://domino.example.com)"
        )
    return f"{parts.scheme}://{parts.netloc}"


def _refuse_missing_owner_or_project(app: DominoApp) -> None:
    # An empty part would yield "{host}//r/..." and every call would 404.
    fields = {"project.ownerUsername": app.owner, "project.name": app.project}
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise DominoDeployError(
            f"App {app.name} has no {' and '.join(missing)} in the Apps API answer, "
            "so the URL of its instance cannot be built"
        )


def _running_instance(client: DominoAppsClient, app: DominoApp) -> AppInstance:
    if app.version_id is None:
        raise DominoDeployError(
            f"App {app.name} has no current version, so no instance to call: "
            "deploy it again"
        )
    instances = client.instances(app.id, app.version_id)
    running = next((i for i in instances if i.status == "Running"), None)
    if running is None:
        seen = ", ".join(i.status for i in instances) or "none"
        raise DominoDeployError(
            f"App {app.name} has no Running instance of its current version "
            f"{app.version_id} to call (instance statuses seen: {seen})"
        )
    return running
