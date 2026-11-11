"""The command-line flags of ``rag-domino-emulation``: placement and App proxy.

Example:
    >>> emulation_parser().parse_args(["--placement", "placement.json"])
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..server import FAKE_DOMINO_PORT
from .proxy_settings import FakeDominoProxySettings, UnauthenticatedResponse


def emulation_parser() -> argparse.ArgumentParser:
    """Build the parser of ``rag-domino-emulation``.

    Returns:
        A parser for ``--host``, ``--port``, the placement and App proxy flags.

    Example:
        >>> emulation_parser().parse_args([]).port
        8899
    """
    parser = argparse.ArgumentParser(
        prog="rag-domino-emulation",
        description="Serve the Domino emulation: the fake Domino 6.2 Apps API, "
        "its App proxy, and Apps placed on machines over SSH.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=FAKE_DOMINO_PORT)
    _add_placement_flags(parser.add_argument_group("Placement"))
    proxy = parser.add_argument_group("App proxy")
    _add_forwarding_flags(proxy, FakeDominoProxySettings())
    _add_identity_flags(proxy, FakeDominoProxySettings())
    return parser


def _add_placement_flags(group: argparse._ArgumentGroup) -> None:
    # Where Apps run in the emulation, and what every App is handed.
    group.add_argument(
        "--placement",
        type=Path,
        metavar="FILE",
        help="JSON of hardware tiers and their machines: each App is started "
        "on a free machine of its tier over SSH (written by Terraform)",
    )
    group.add_argument(
        "--project-env",
        type=Path,
        metavar="FILE",
        help="KEY=VALUE lines exported to every placed App (needs --placement)",
    )


def _add_forwarding_flags(
    proxy: argparse._ArgumentGroup, defaults: FakeDominoProxySettings
) -> None:
    # Where a request for an App goes, and what of its path it keeps.
    proxy.add_argument(
        "--upstream",
        action="append",
        default=[],
        type=_upstream,
        metavar="NAME=URL",
        help="where the App named NAME runs (repeatable; wins over its machine)",
    )
    proxy.add_argument(
        "--keep-prefix",
        action="store_true",
        help="forward the Domino path prefix to the App instead of stripping it",
    )
    proxy.add_argument(
        "--forward-timeout",
        type=float,
        metavar="SECONDS",
        default=defaults.forward_timeout_seconds,
    )


def _add_identity_flags(
    proxy: argparse._ArgumentGroup, defaults: FakeDominoProxySettings
) -> None:
    # Who gets through, and how the App is told who they are.
    proxy.add_argument(
        "--no-identity-jwt",
        dest="identity_jwt",
        action="store_false",
        help="secure identity propagation off: only domino-username is passed",
    )
    proxy.add_argument(
        "--unauthenticated",
        default=defaults.unauthenticated.value,
        choices=[mode.value for mode in UnauthenticatedResponse],
        help="answer to no credential: 401, or a 302 to the login page",
    )


def _upstream(text: str) -> tuple[str, str]:
    # argparse prints an ArgumentTypeError's message as is, the value included.
    name, _, url = text.partition("=")
    if not name or not url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError(
            f"--upstream {text!r}: expected NAME=URL, "
            "e.g. rag-reranker=http://127.0.0.1:9001"
        )
    return name, url
