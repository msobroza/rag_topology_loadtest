"""End-to-end tests of the ``rag-deploy`` CLI: ``main`` over the in-process fake
Domino (status, url, deploy, per-behaviour failures, --dry-run, deploy.env).
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from domino_deploy_helpers import APPS_PATH, APPS_URL, DEPLOY_ENV, add_app
from rag_load_test.deploy.cli import main
from rag_load_test.deploy.fake_domino import FakeDominoApps, create_fake_domino

PROXY = "http://proxy.test:8899"


# --- rag-deploy end to end: main over the fake Domino ---


@pytest.fixture
def domino_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Workspace's environment, run from a repository root with no deploy.env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "domino").mkdir()
    (tmp_path / "domino" / "app.sh").write_text("#!/bin/bash\n")
    for name in ("RAG_DEPLOY_TOPOLOGY", "RAG_DEPLOY_PROJECT_ID", "RAG_DEPLOY_ES_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOMINO_API_PROXY", PROXY)
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-1")
    monkeypatch.setenv("RAG_DEPLOY_ENVIRONMENT_ID", "env-1")


@pytest.fixture
def no_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on any request through a real transport (TestClient has its own)."""

    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"real HTTP request sent: {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


def launched() -> str:
    """The tail of every entry script when rag-deploy runs from this directory."""
    return f"exec bash {shlex.quote(str(Path.cwd() / 'domino' / 'app.sh'))}"


RunMain = Callable[..., tuple[int, str, str]]


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


def answering(
    app: ASGIApp, responses: dict[str, Response]
) -> tuple[ASGIApp, list[Request]]:
    """``app``, except that the paths in ``responses`` get those canned answers.

    Returns the wrapped app and the requests that got a canned answer, in order.
    """
    seen: list[Request] = []

    async def wrapped(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in responses:
            return await app(scope, receive, send)
        seen.append(Request(scope))
        await responses[scope["path"]](scope, receive, send)

    return wrapped, seen


@pytest.mark.parametrize("variable", ["DOMINO_API_PROXY", "DOMINO_PROJECT_ID"])
def test_outside_a_workspace_the_missing_variable_is_named(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.delenv(variable)
    code, _, err = run_main(["status"], None)
    assert code == 2
    assert f"{variable} is not set" in err
    assert "runs in a Domino Workspace" in err


def test_other_failures_show_the_status_and_the_start_of_the_body(
    run_main: RunMain,
) -> None:
    code, _, err = run_main(
        ["deploy", "all"], PlainTextResponse("x" * 300, status_code=500)
    )
    assert code == 2
    assert err.endswith(f"returned HTTP 500: {'x' * 200}\n")


READY_BODY = {
    "status": "ready",
    "checks": [
        {"name": "embedder", "ok": True, "detail": "ovms:bge-small-en-v1.5"},
        {"name": "vector_store", "ok": False, "detail": "ConnectError"},
    ],
}
RUN_HEADERS = {
    "X-RAG-Topology": "split-reranker",
    "X-RAG-Run-Label": "split-reranker-fastapi",
}


def test_status_lists_the_apps_then_the_workflow_readiness(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "split-reranker")
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow")
    readyz = JSONResponse(READY_BODY, status_code=503, headers=RUN_HEADERS)
    token = PlainTextResponse("tok-1\n")
    canned = {"/access-token": token, "/apps/app-1/readyz": readyz}
    fake, seen = answering(create_fake_domino(apps), canned)
    code, out, _ = run_main(["status"], fake)
    assert code == 0
    assert out == (
        "rag-reranker   not deployed\n"
        f"rag-workflow   Running        {APPS_URL}/app-1\n"
        "readyz: HTTP 503, X-RAG-Topology: split-reranker, "
        "X-RAG-Run-Label: split-reranker-fastapi\n"
        "  embedder: ok ovms:bge-small-en-v1.5\n"
        "  vector_store: FAIL ConnectError\n"
    )
    assert [str(request.url) for request in seen] == [
        "http://localhost:8899/access-token",
        f"{APPS_URL}/app-1/readyz",
    ]
    assert seen[1].headers["Authorization"] == "Bearer tok-1"


def test_status_still_succeeds_when_the_run_token_is_unavailable(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow")
    no_token = {"/access-token": PlainTextResponse("no token", status_code=500)}
    fake, _ = answering(create_fake_domino(apps), no_token)
    code, out, _ = run_main(["status"], fake)
    assert code == 0
    assert out == (
        f"rag-workflow   Running        {APPS_URL}/app-1\n"
        "readyz: not checked (run token from http://localhost:8899/access-token: "
        "HTTP 500)\n"
    )
    assert ("GET", "/apps/app-1/readyz") not in apps.received


def test_status_skips_readiness_when_the_workflow_app_is_not_running(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    apps = FakeDominoApps([None])
    add_app(apps, "rag-workflow")
    code, out, _ = run_main(["status"], create_fake_domino(apps))
    assert (code, out) == (0, f"rag-workflow   no instance    {APPS_URL}/app-1\n")
    assert ("GET", "/access-token") not in apps.received


def test_url_prints_only_the_app_url(run_main: RunMain) -> None:
    listed = {
        "items": [{"id": "app-1", "name": "rag-workflow", "url": "https://d.test/a/"}]
    }
    fake, _ = answering(create_fake_domino(), {APPS_PATH: JSONResponse(listed)})
    assert run_main(["url", "workflow"], fake) == (0, "https://d.test/a\n", "")


def test_url_of_an_app_not_deployed_is_an_error(run_main: RunMain) -> None:
    code, out, err = run_main(["url", "reranker"], create_fake_domino())
    assert (code, out) == (2, "")
    assert err.endswith(
        "rag-deploy: error: App rag-reranker is not deployed in project proj-1: "
        "run make deploy-reranker\n"
    )


def test_deploy_requires_the_environment_id(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_DEPLOY_ENVIRONMENT_ID")
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert err.endswith(
        "rag-deploy: error: RAG_DEPLOY_ENVIRONMENT_ID is not set: it is the id of "
        "the Domino compute environment the Apps run in\n"
    )
    assert apps.received == []


def test_deploy_outside_the_repository_root_publishes_nothing(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert err.endswith(
        f"rag-deploy: error: {Path.cwd()} has no domino/app.sh: run rag-deploy from "
        "the repository root (for example /repos/<project>), the directory the "
        "Apps run it from\n"
    )
    assert apps.received == []


def test_ovms_deploy_checks_the_bundle_exists_before_publishing(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_MODEL_SERVER", "ovms")
    monkeypatch.setenv("RAG_DEPLOY_OVMS_BUNDLE", str(tmp_path / "missing.tar.gz"))
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert err.endswith(
        f"rag-deploy: error: RAG_DEPLOY_OVMS_BUNDLE={tmp_path / 'missing.tar.gz'} "
        "does not exist: set it to the OVMS bundle folder (bin/ovms, lib/) or its "
        ".tar.gz, at a path the Apps can read\n"
    )
    assert apps.received == []


def test_ovms_deploy_publishes_when_the_bundle_exists(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_MODEL_SERVER", "ovms")
    monkeypatch.setenv("RAG_DEPLOY_OVMS_BUNDLE", str(tmp_path))
    apps = FakeDominoApps(["Running"])
    code, _, _ = run_main(["deploy", "reranker"], create_fake_domino(apps))
    assert code == 0
    assert [body.name for body in apps.creates] == ["rag-reranker"]
    assert str(apps.creates[0].version.entryScript).endswith(launched())


# --- rag-deploy per fake Domino behaviour: what a user would read ---


def deployed_lines(count: int) -> str:
    names = ["rag-embedder", "rag-reranker", "rag-workflow"][:count]
    return "".join(
        f"{name}: Running {APPS_URL}/app-{n}\n" for n, name in enumerate(names, 1)
    )


@pytest.mark.parametrize("behaviour", ["starts", "slow"])
def test_deploy_all_succeeds_when_the_apps_start(
    run_main: RunMain, behaviour: str
) -> None:
    apps = FakeDominoApps.from_behaviour(behaviour)
    code, out, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert (code, out, err) == (0, deployed_lines(3), "")


def test_deploy_all_stops_at_the_first_app_that_fails(run_main: RunMain) -> None:
    apps = FakeDominoApps.from_behaviour("fails")
    code, out, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert (code, out) == (2, "")
    assert err.endswith(
        f"rag-deploy: error: App rag-embedder is Failed; see its logs at "
        f"{APPS_URL}/app-1\n"
    )
    assert [body.name for body in apps.creates] == ["rag-embedder"]


def test_deploy_all_says_to_start_an_app_domino_did_not_start(
    run_main: RunMain,
) -> None:
    apps = FakeDominoApps.from_behaviour("never-starts")
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert err.endswith(
        "rag-deploy: error: Domino created App rag-embedder but did not start it: "
        f"start it once from the Apps page ({APPS_URL}/app-1); the Domino 6.2 Apps "
        "API has no start call\n"
    )


def test_a_403_shows_the_errors_text(run_main: RunMain) -> None:
    apps = FakeDominoApps.from_behaviour("forbidden")
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert err.endswith(
        "rag-deploy: error: Domino Apps API GET /api/apps/beta/apps returned HTTP "
        "403: Forbidden by the fake Domino (behaviour: forbidden)\n"
    )
    assert apps.creates == []


def test_status_after_a_deploy_shows_the_run_headers_the_app_reports(
    run_main: RunMain,
) -> None:
    fake = create_fake_domino(FakeDominoApps.from_behaviour("starts"))
    assert run_main(["deploy", "all"], fake)[0] == 0
    code, out, _ = run_main(["status"], fake)
    assert code == 0
    assert out == (
        f"rag-embedder   Running        {APPS_URL}/app-1\n"
        f"rag-reranker   Running        {APPS_URL}/app-2\n"
        f"rag-workflow   Running        {APPS_URL}/app-3\n"
        "readyz: HTTP 200, X-RAG-Topology: split-all, "
        "X-RAG-Run-Label: split-all-fastapi\n"
        "  fake_domino: ok answered by the fake\n"
    )


# --- rag-deploy --dry-run: the same code against an in-process fake Domino ---


def dry_run_report(env_id: str, tier_line: str = "") -> str:
    tail = launched()
    model_servers = "".join(
        f"\nrag-{part}\n  environmentId: {env_id}\n{tier_line}  entryScript:\n"
        f"    export RAG_ROLE=fastapi-{part}\n    export RAG_BOOTSTRAP=1\n    {tail}\n"
        for part in ("embedder", "reranker")
    )
    return (
        "\nDry run: nothing was sent to Domino. rag-deploy would publish:\n"
        f"{model_servers}"
        f"\nrag-workflow\n  environmentId: {env_id}\n{tier_line}  entryScript:\n"
        "    export RAG_ROLE=workflow\n    export RAG_TOPOLOGY=split-all\n"
        f"    export RAG_OVMS_RERANK_URL={APPS_URL}/app-2\n"
        f"    export RAG_OVMS_EMBEDDINGS_URL={APPS_URL}/app-1\n"
        "    export RAG_RUN_LABEL=split-all-fastapi\n"
        f"    export RAG_BOOTSTRAP=1\n    {tail}\n"
    )


def test_dry_run_deploy_needs_no_workspace_and_prints_each_entry_script(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY")
    monkeypatch.delenv("DOMINO_PROJECT_ID")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    assert out == deployed_lines(3) + dry_run_report("env-1")


def test_dry_run_shows_the_hardware_tier_when_set(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_HARDWARE_TIER_ID", "tier-8")
    code, out, _ = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 0
    assert out.endswith(dry_run_report("env-1", "  hardwareTierId: tier-8\n"))


def test_dry_run_still_requires_the_environment_id(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_DEPLOY_ENVIRONMENT_ID")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert err.endswith("RAG_DEPLOY_ENVIRONMENT_ID is not set: it is the id of "
        "the Domino compute environment the Apps run in\n")  # fmt: skip


def test_dry_run_still_refuses_secrets_in_deploy_env(run_main: RunMain) -> None:
    Path("deploy.env").write_text("RAG_ES_API_KEY=secret\n")
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, out) == (2, "")
    assert "RAG_ES_API_KEY is set in deploy.env" in err


def test_dry_run_still_requires_the_repository_root(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2 and "has no domino/app.sh" in err


def test_dry_run_still_checks_the_ovms_bundle(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_MODEL_SERVER", "ovms")
    monkeypatch.setenv("RAG_DEPLOY_OVMS_BUNDLE", "/no/such/bundle")
    code, _, err = run_main(["--dry-run", "deploy", "all"], None)
    assert code == 2 and "RAG_DEPLOY_OVMS_BUNDLE=/no/such/bundle does not exist" in err


def test_dry_run_status_lists_nothing_deployed(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY")
    code, out, _ = run_main(["--dry-run", "status"], None)
    assert (code, out) == (
        0,
        "rag-embedder   not deployed\n"
        "rag-reranker   not deployed\n"
        "rag-workflow   not deployed\n",
    )


# --- deploy.env ---


def test_deploy_passes_deploy_env_to_every_app(run_main: RunMain) -> None:
    Path("deploy.env").write_text(DEPLOY_ENV)  # run_main runs in an empty tmp dir
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 0, err
    scripts = [str(body.version.entryScript) for body in apps.creates]
    assert len(scripts) == 2  # split-reranker, from RAG_DEPLOY_TOPOLOGY in the file
    for entry in scripts:
        assert entry.startswith(
            "export RAG_EMBEDDER_MODEL=/mnt/data/models/bge-small\n"
            "export RAG_ES_URL=https://es.test:9200\n"
            "export RAG_RERANKER_MODEL='/mnt/data/models/bge reranker'\n"
            "export RAG_ROLE="
        )
        assert entry.endswith(f"export RAG_BOOTSTRAP=1\n{launched()}")


def test_deploy_refuses_an_es_api_key_in_deploy_env(run_main: RunMain) -> None:
    Path("deploy.env").write_text("RAG_ES_API_KEY=secret\n")
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2 and "RAG_ES_API_KEY is set in deploy.env" in err
    assert apps.creates == []
