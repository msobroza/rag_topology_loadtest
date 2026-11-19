"""Serve an app with Uvicorn: the app object with one worker, its factory with more.

Uvicorn 0.54.0 logs "You must pass the application as an import string to
enable 'reload' or 'workers'" and exits when ``workers`` is above 1 and the app
is an object. Above 1, each worker imports the factory and builds the app
itself, from the settings and from what the command exported for it, since a
worker never sees the command's arguments.

Example:
    >>> address = {"host": "0.0.0.0", "port": 8888}
    >>> run_uvicorn(app, "pkg.module:app_from_environment",
    ...             arguments={**address, **UvicornSettings().run_arguments()},
    ...             exports={"QP_SERVE_DOMINO": "1"})
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping

import uvicorn
from starlette.types import ASGIApp


def run_uvicorn(
    app: ASGIApp,
    factory: str,
    *,
    arguments: Mapping[str, str | int | None],
    exports: Mapping[str, str],
    environ: MutableMapping[str, str] = os.environ,
    run: Callable[..., None] = uvicorn.run,
) -> None:
    """Serve ``app``, or with several workers the app ``factory`` builds in each.

    Args:
        app: The app the command built, served as it is by one worker.
        factory: The import string of a function building the same app from
            the environment, such as ``pkg.module:app_from_environment``.
        arguments: ``uvicorn.run``'s keywords: ``host`` and ``port``, and those
            of ``UvicornSettings``, the workers count among them.
        exports: What a worker cannot read from the command's arguments, as
            variables; exported only when there are workers to read them.
        environ: The environment the workers inherit.
        run: Serves the app; ``uvicorn.run`` unless a test replaces it.

    Example:
        >>> run_uvicorn(app, "m:f", arguments={"port": 8888, "workers": 1},
        ...             exports={})
    """
    if (arguments.get("workers") or 1) == 1:
        run(app, **arguments)
        return
    environ.update(exports)
    run(factory, factory=True, **arguments)
