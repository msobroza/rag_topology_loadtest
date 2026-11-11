"""The start script the fake runs on an App's machine, around its ``entryScript``.

It exports Domino's run variables, then the project environment variables, then
runs the version's ``entryScript`` unchanged, so the script's own exports win.
Documented by Domino: ``DOMINO_API_PROXY``, ``DOMINO_HARDWARE_TIER_ID``,
``DOMINO_PROJECT_ID`` and ``DOMINO_RUN_HOST_PATH`` are set in a run.
Assumed: ``DOMINO_APP_ID`` and ``DOMINO_APP_VERSION_ID`` are set in an App's
run; ``DOMINO_RUN_HOST_PATH`` is the instance's legacy path prefix, ending in
``/``. Emulation only: ``DOMINO_API_PROXY`` points at the fake through a port
each machine forwards.

Example:
    >>> compose_start_script(run, "http://localhost:8899", {}, "exec bash app.sh")
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import NamedTuple


class DominoRunVariables(NamedTuple):
    """What Domino tells an App's run about itself, besides the API proxy.

    Attributes:
        hardware_tier_id: The tier the App runs on (the default's id if the
            version named none).
        project_id: The App's project.
        app_id: The App.
        app_version_id: The version being started.
        run_host_path: The instance's path prefix, ending in ``/``.
    """

    hardware_tier_id: str
    project_id: str
    app_id: str
    app_version_id: str
    run_host_path: str


def compose_start_script(
    run: DominoRunVariables,
    api_proxy_url: str,
    project_env: Mapping[str, str],
    entry_script: str,
) -> str:
    r"""Compose the bash script that starts one App version on its machine.

    Args:
        run: Domino's run variables for this version.
        api_proxy_url: ``DOMINO_API_PROXY``, e.g. ``"http://localhost:8899"``.
        project_env: The project environment variables, in order.
        entry_script: The version's ``entryScript``, run unchanged last.

    Returns:
        The script: one shell-quoted ``export`` per variable, then the entry
        script.

    Example:
        >>> compose_start_script(run, "http://localhost:8899", {"K": "a b"}, "true")
        "export DOMINO_API_PROXY=http://localhost:8899\n...export K='a b'\ntrue\n"
    """
    domino = {
        "DOMINO_API_PROXY": api_proxy_url,
        "DOMINO_HARDWARE_TIER_ID": run.hardware_tier_id,
        "DOMINO_PROJECT_ID": run.project_id,
        "DOMINO_APP_ID": run.app_id,
        "DOMINO_APP_VERSION_ID": run.app_version_id,
        "DOMINO_RUN_HOST_PATH": run.run_host_path,
    }
    exports = [
        f"export {name}={shlex.quote(value)}"
        for name, value in [*domino.items(), *project_env.items()]
    ]
    return "\n".join([*exports, entry_script, ""])
