"""Fixtures of the ``qp-deploy`` CLI tests: ``main`` run in a fake Workspace.

Loaded by ``pytest_plugins = ["domino_deploy_fixtures"]`` (never imported by
name, so pytest can rewrite its asserts) in ``test_qp_deploy.py`` and
``test_domino_deploy_dry_run.py``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from starlette.types import ASGIApp

from domino_deploy_helpers import ENV_1, PROXY, RunMain
from query_processing.deploy.qp_deploy import main


@pytest.fixture
def domino_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Workspace's environment, run from a use case's directory whose deploy.yaml
    only sets the environment, beside the toolkit's in a repository."""
    repository = tmp_path / "repository"
    launch_script = repository / "query_processing" / "domino" / "app.sh"
    launch_script.parent.mkdir(parents=True)
    launch_script.write_text("#!/bin/bash\n")
    use_case = repository / "my_use_case"
    use_case.mkdir()
    monkeypatch.chdir(use_case)
    (use_case / "deploy.yaml").write_text(ENV_1)
    for name in (
        "TOPOLOGY",
        "PROJECT_ID",
        "FILE",
        "OVMS_BUNDLE",
        "APP_NAME_PREFIX",
    ):
        monkeypatch.delenv(f"QP_DEPLOY_{name}", raising=False)
    monkeypatch.delenv("QP_DEPLOY_DOMINO_HOST", raising=False)
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
    del domino_env, no_real_http  # requested for what they set up, not their value

    def run(argv: list[str], fake_domino: ASGIApp | None) -> tuple[int, str, str]:
        code = 0
        try:
            main(argv, fake_domino=fake_domino)
        except SystemExit as exc:
            code = int(exc.code or 0)
        out, err = capsys.readouterr()
        return code, out, err

    return run
