"""End-to-end tests of the ``rag-deploy`` CLI: ``main`` over the in-process fake
Domino (status, url, deploy, per-behaviour failures, deploy.yaml, the App id
cache).
Dry runs: test_domino_deploy_dry_run.py. The fixtures are in domino_deploy_fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from domino_deploy_helpers import (
    APPS_URL,
    CACHE_FILE,
    DEPLOY_YAML,
    ENV_1,
    INSTANCES_URL,
    NO_ENVIRONMENT,
    RunMain,
    add_app,
    deployed_lines,
    launched,
)
from fake_domino.server import create_fake_domino
from fake_domino.state import FakeDominoApps

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main
READYZ = "/fake-owner/fake-project/r/notebookSession/app-1-v1-i/readyz"


# --- rag-deploy end to end: main over the fake Domino ---


def answering(
    app: ASGIApp, responses: dict[str, Response]
) -> tuple[ASGIApp, list[Request]]:
    """``app``, except that the paths in ``responses`` get those canned answers.

    Returns the wrapped app and the requests that got a canned answer, in order.
    """
    seen: list[Request] = []

    async def wrapped(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in responses:
            await app(scope, receive, send)
            return
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
    canned = {"/access-token": token, READYZ: readyz}
    fake, seen = answering(create_fake_domino(apps), canned)
    code, out, _ = run_main(["status"], fake)
    assert code == 0
    assert out == (
        "rag-reranker   not deployed\n"
        f"rag-workflow   Running        {INSTANCES_URL}/app-1-v1-i\n"
        "readyz: HTTP 503, X-RAG-Topology: split-reranker, "
        "X-RAG-Run-Label: split-reranker-fastapi\n"
        "  embedder: ok ovms:bge-small-en-v1.5\n"
        "  vector_store: FAIL ConnectError\n"
    )
    assert [str(request.url) for request in seen] == [
        "http://localhost:8899/access-token",
        f"{INSTANCES_URL}/app-1-v1-i/readyz",
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
        f"rag-workflow   Running        {INSTANCES_URL}/app-1-v1-i\n"
        "readyz: not checked (run token from http://localhost:8899/access-token: "
        "HTTP 500)\n"
    )
    assert ("GET", READYZ) not in apps.received


def test_status_asks_for_the_run_token_where_the_settings_say(
    run_main: RunMain,
) -> None:
    Path("deploy.yaml").write_text(
        ENV_1 + "topology: monolith\nrun_token_url: http://tokens.test/t\n"
    )
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow")
    canned = {"/t": PlainTextResponse("no token", status_code=500)}
    fake, seen = answering(create_fake_domino(apps), canned)
    code, out, _ = run_main(["status"], fake)
    assert code == 0
    assert "run token from http://tokens.test/t: HTTP 500" in out
    assert [str(request.url) for request in seen] == ["http://tokens.test/t"]


def test_status_skips_readiness_when_the_workflow_app_is_not_running(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    apps = FakeDominoApps([None])
    add_app(apps, "rag-workflow")
    code, out, _ = run_main(["status"], create_fake_domino(apps))
    assert (code, out) == (0, f"rag-workflow   no instance    {APPS_URL}/app-1\n")
    assert ("GET", "/access-token") not in apps.received


def test_url_prints_only_where_the_app_answers(run_main: RunMain) -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow")
    expected = f"{INSTANCES_URL}/app-1-v1-i\n"
    assert run_main(["url", "workflow"], create_fake_domino(apps)) == (0, expected, "")


def test_url_takes_the_host_from_domino_host(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_DOMINO_HOST", "https://domino.example")
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow")
    code, out, _ = run_main(["url", "workflow"], create_fake_domino(apps))
    assert (code, out) == (
        0,
        "https://domino.example/fake-owner/fake-project/r/notebookSession/app-1-v1-i\n",
    )


def test_url_of_an_app_with_no_running_instance_is_an_error(
    run_main: RunMain,
) -> None:
    apps = FakeDominoApps(["Pending"])
    add_app(apps, "rag-workflow")
    code, out, err = run_main(["url", "workflow"], create_fake_domino(apps))
    assert (code, out) == (2, "")
    assert err.endswith(
        "rag-deploy: error: App rag-workflow has no Running instance of its current "
        "version app-1-v1 to call (instance statuses seen: Pending)\n"
    )


def test_url_of_an_app_not_deployed_is_an_error(run_main: RunMain) -> None:
    code, out, err = run_main(["url", "reranker"], create_fake_domino())
    assert (code, out) == (2, "")
    assert err.endswith(
        "rag-deploy: error: App rag-reranker is not deployed in project proj-1: "
        "run make deploy-reranker\n"
    )


def test_deploy_requires_the_environment_id(run_main: RunMain) -> None:
    Path("deploy.yaml").unlink()
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert err.endswith(NO_ENVIRONMENT)
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
        f"rag-deploy: error: ovms_bundle={tmp_path / 'missing.tar.gz'} "
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
        f"rag-embedder   Running        {INSTANCES_URL}/app-1-v1-i\n"
        f"rag-reranker   Running        {INSTANCES_URL}/app-2-v1-i\n"
        f"rag-workflow   Running        {INSTANCES_URL}/app-3-v1-i\n"
        "readyz: HTTP 200, X-RAG-Topology: split-all, "
        "X-RAG-Run-Label: split-all-fastapi\n"
        "  fake_domino: ok answered by the fake\n"
    )


# --- deploy.yaml ---


def test_deploy_passes_each_apps_settings_to_it(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(DEPLOY_YAML)  # run_main runs in an empty tmp dir
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 0, err
    scripts = [str(body.version.entryScript) for body in apps.creates]
    assert len(scripts) == 2  # split-reranker, from `topology` in the file
    assert scripts[0].startswith(
        "export RAG_ES_URL=https://es.test:9200\n"
        "export RAG_MODEL_WORKERS=2\n"
        "export RAG_RERANKER_MODEL='/mnt/data/models/bge reranker'\n"
        "export RAG_ROLE=fastapi-reranker\n"
    )
    assert scripts[1].startswith(
        "export RAG_ES_URL=https://es.test:9200\n"
        "export RAG_RERANKER_MODEL='/mnt/data/models/bge reranker'\n"
        "export RAG_ROLE=workflow\n"
    )
    for entry in scripts:
        assert entry.endswith(f"export RAG_BOOTSTRAP=1\n{launched()}")


def test_deploy_refuses_an_es_api_key_in_deploy_yaml(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "  settings: {RAG_ES_API_KEY: secret}\n")
    apps = FakeDominoApps(["Running"])
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert code == 2
    assert "RAG_ES_API_KEY is set in deploy.yaml" in err
    assert apps.creates == []


def test_a_misspelt_key_in_deploy_yaml_is_named(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "apps: {embedder: {hardware_tier: x}}\n")
    code, _, err = run_main(["status"], create_fake_domino())
    assert code == 2
    assert "apps.embedder.hardware_tier" in err


# --- the App id cache ---


def test_a_second_deploy_redeploys_the_same_apps(run_main: RunMain) -> None:
    apps = FakeDominoApps(["Running"])
    fake = create_fake_domino(apps)
    assert run_main(["deploy", "all"], fake)[0] == 0
    assert run_main(["deploy", "all"], fake)[0] == 0
    assert len(apps.creates) == 3
    assert [app_id for app_id, _ in apps.new_versions] == ["app-1", "app-2", "app-3"]
    assert json.loads(Path(CACHE_FILE).read_text()) == {
        "proj-1": {
            "rag-embedder": "app-1",
            "rag-reranker": "app-2",
            "rag-workflow": "app-3",
        }
    }


def test_a_cache_file_that_cannot_be_read_is_named(run_main: RunMain) -> None:
    Path(CACHE_FILE).write_text("{oops")
    code, _, err = run_main(["status"], create_fake_domino())
    assert code == 2
    assert f"{CACHE_FILE} is not an App id cache" in err
