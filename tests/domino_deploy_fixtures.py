"""Fixtures of the ``rag-deploy`` CLI tests: ``main`` run in a fake Workspace.

Loaded by ``pytest_plugins = ["domino_deploy_fixtures"]`` (never imported by
name, so pytest can rewrite its asserts) in ``test_rag_deploy_cli.py`` and
``test_domino_deploy_dry_run.py``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from starlette.types import ASGIApp

from domino_deploy_helpers import ENV_1, PROXY, RunMain
from rag_load_test.deploy.cli import main


@pytest.fixture
def domino_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Workspace's environment, run from a repository root whose deploy.yaml
    only sets the environment."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "domino").mkdir()
    (tmp_path / "domino" / "app.sh").write_text("#!/bin/bash\n")
    (tmp_path / "deploy.yaml").write_text(ENV_1)
    for name in ("TOPOLOGY", "PROJECT_ID", "FILE", "MODEL_SERVER", "OVMS_BUNDLE"):
        monkeypatch.delenv(f"RAG_DEPLOY_{name}", raising=False)
    monkeypatch.delenv("RAG_DEPLOY_DOMINO_HOST", raising=False)
    monkeypatch.setenv("DOMINO_API_PROXY", PROXY)
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-1")


@pytest.fixture
def no_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on any request through a real transport (TestClient has its own)."""

    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"real HTTP request sent: {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture
def run_main(
    capsys: pytest.CaptureFixture[str], domino_env: None, no_real_http: None
) -> RunMain:
    """Runs ``main`` against ``fake_domino``; returns (exit code, stdout, stderr)."""

    def run(argv: list[str], fake_domino: ASGIApp | None) -> tuple[int, str, str]:
        code = 0
        try:
            main(argv, fake_domino=fake_domino)
        except SystemExit as exc:
            code = int(exc.code or 0)
        out, err = capsys.readouterr()
        return code, out, err

    return run
