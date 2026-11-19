"""``run_uvicorn``: one worker serves the app object, several the factory's import string.

Uvicorn exits on an app object with more than one worker, so above one the
app is named by its factory, which each worker calls; what a worker cannot
read from the command line is exported for it first.
"""

from __future__ import annotations

from typing import Any

from query_processing.config.apps.uvicorn_settings import UvicornSettings
from query_processing.infrastructure.http.uvicorn_run import run_uvicorn

APP = object()  # an app object, as main builds it
FACTORY = "m:f"
Call = tuple[tuple[Any, ...], dict[str, Any]]


def run_recording(
    workers: int, exports: dict[str, str], environ: dict[str, str]
) -> list[Call]:
    """Call ``run_uvicorn`` at ``workers``; return what Uvicorn was asked."""
    calls: list[Call] = []
    settings = UvicornSettings(_env_file=None, workers=workers)
    run_uvicorn(
        APP,
        FACTORY,
        arguments={"host": "0.0.0.0", "port": 8888, **settings.run_arguments()},
        exports=exports,
        environ=environ,
        run=lambda *a, **k: calls.append((a, k)),
    )
    return calls


def test_one_worker_runs_the_app_object_as_today() -> None:
    environ: dict[str, str] = {}

    [(args, kwargs)] = run_recording(1, {"QP_SERVE_DOMINO": "1"}, environ)

    assert args == (APP,)
    assert "factory" not in kwargs
    assert kwargs["workers"] == 1
    assert (kwargs["host"], kwargs["port"]) == ("0.0.0.0", 8888)
    assert environ == {}  # no worker to hand anything to


def test_several_workers_run_the_import_string() -> None:
    environ = {"QP_SERVE_DOMINO": "1", "PATH": "/bin"}

    [(args, kwargs)] = run_recording(2, {"QP_SERVE_DOMINO": "0"}, environ)

    assert args == (FACTORY,)
    assert kwargs["factory"] is True
    assert kwargs["workers"] == 2
    # A worker reads what main read from its arguments, not a stale value.
    assert environ == {"QP_SERVE_DOMINO": "0", "PATH": "/bin"}
