"""Unit tests for ``rag_load_test.domino_deploy`` against a fake Domino Apps API.

``FakeDominoAppsApi`` stands in for Domino 6.2's beta Apps REST API behind
``httpx.MockTransport``: it stores Apps, answers the four documented operations
with the documented shapes, records every request and returns scripted instance
statuses, so no live Domino and no real HTTP is needed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from rag_load_test.domino_apps import (
    NO_INSTANCE_GRACE_S,
    WAIT_TIMEOUT_S,
    DominoAppsClient,
    DominoDeployError,
)
from rag_load_test.domino_deploy import (
    DeploySettings,
    apps_for_topology,
    deploy,
    entry_script,
    load_app_settings,
    main,
    publish_app,
)

PROXY = "http://proxy.test:8899"
APPS_PATH = "/api/apps/beta/apps"


@dataclass(frozen=True)
class RecordedCall:
    method: str
    path: str
    params: dict[str, str]
    body: object


class FakeDominoAppsApi:
    """The four Apps operations over ``httpx.MockTransport``, with scripted statuses.

    Each ``GET /apps/{id}`` takes the next entry of ``statuses`` (the last one
    repeats); ``None`` means the App has no instance. ``error`` answers every
    request with that response instead. ``stale_gets`` makes that many GETs still
    report the App's previous version as Running (a redeploy not yet switched).
    """

    def __init__(self, *statuses: str | None) -> None:
        self.statuses = list(statuses) or ["Running"]
        self.apps: dict[str, dict[str, object]] = {}
        self.calls: list[RecordedCall] = []
        self.error: httpx.Response | None = None
        self.stale_gets = 0
        self._previous_version: dict[str, str] = {}

    def add_app(self, name: str, project_id: str = "proj-1") -> dict[str, object]:
        app_id = f"app-{len(self.apps) + 1}"
        app: dict[str, object] = {
            "id": app_id,
            "name": name,
            "projectId": project_id,
            "entryPoint": "domino/app.sh",
            "url": f"https://domino.test/apps/{app_id}",
            "visibility": "AUTHENTICATED",
            "currentVersion": {"id": f"{app_id}-v1"},
        }
        self.apps[app_id] = app
        return app

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        params = dict(request.url.params)
        self.calls.append(RecordedCall(request.method, request.url.path, params, body))
        if self.error is not None:
            return self.error
        return self._route(request.method, request.url.path, params, body)

    def _route(
        self, method: str, path: str, params: dict[str, str], body: object
    ) -> httpx.Response:
        parts = path.removeprefix(APPS_PATH).strip("/").split("/")
        if method == "GET" and parts == [""]:
            return httpx.Response(200, json=self._list(params))
        if method == "POST" and parts == [""]:
            return httpx.Response(200, json=self._create(body))
        if method == "POST" and parts[1:] == ["versions"]:
            return httpx.Response(200, json=self._new_version(parts[0]))
        if method == "GET" and len(parts) == 1 and parts[0] in self.apps:
            return httpx.Response(200, json=self._get(parts[0]))
        return httpx.Response(404)

    def _list(self, params: dict[str, str]) -> dict[str, object]:
        # Domino may match names loosely; this fake returns prefix matches too.
        items = [
            app
            for app in self.apps.values()
            if app["projectId"] == params["projectId"]
            and str(app["name"]).startswith(params["name"])
        ]
        return {"items": items, "metadata": {"totalCount": len(items)}}

    def _create(self, body: object) -> dict[str, object]:
        assert isinstance(body, dict)
        return self.add_app(body["name"], body["projectId"])

    def _new_version(self, app_id: str) -> dict[str, object]:
        app = self.apps[app_id]
        old = app["currentVersion"]
        assert isinstance(old, dict)
        self._previous_version[app_id] = old["id"]
        version = {"id": f"{app_id}-v{int(old['id'].rsplit('v', 1)[1]) + 1}"}
        app["currentVersion"] = version
        return version

    def _get(self, app_id: str) -> dict[str, object]:
        app = dict(self.apps[app_id])
        version = dict(app["currentVersion"])  # type: ignore[call-overload]
        if self.stale_gets:
            self.stale_gets -= 1
            stale = {"id": self._previous_version[app_id]}
            app["currentVersion"] = {**stale, "currentInstance": _instance("Running")}
            return app
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status is not None:
            version["currentInstance"] = _instance(status)
        app["currentVersion"] = version
        return app

    def calls_to(self, method: str) -> list[RecordedCall]:
        return [call for call in self.calls if call.method == method]


def _instance(status: str) -> dict[str, str]:
    return {"id": "inst-1", "status": status}


def make_client(
    fake: FakeDominoAppsApi, project_id: str = "proj-1"
) -> DominoAppsClient:
    http = httpx.Client(base_url=PROXY, transport=fake.transport())
    return DominoAppsClient(http, project_id)


def no_sleep(seconds: float) -> None:
    pass


@pytest.mark.parametrize(
    ("topology", "apps"),
    [
        ("split-all", ["embedder", "reranker", "workflow"]),
        ("split-reranker", ["reranker", "workflow"]),
        ("monolith", ["workflow"]),
    ],
)
def test_each_topology_needs_its_apps_in_deploy_order(
    topology: str, apps: list[str]
) -> None:
    assert apps_for_topology(topology) == apps  # type: ignore[arg-type]


def make_settings(**overrides: object) -> DeploySettings:
    """DeploySettings without reading a stray ``deploy.env``."""
    values: dict[str, object] = {
        "project_id": "proj-1",
        "environment_id": "env-1",
        **overrides,
    }
    return DeploySettings(_env_file=None, **values)  # type: ignore[call-arg]


BOOTSTRAP_AND_EXEC = "export RAG_BOOTSTRAP=1\nexec bash domino/app.sh"


@pytest.mark.parametrize("part", ["embedder", "reranker"])
def test_fastapi_model_server_script_exports_its_role_and_bootstrap(part: str) -> None:
    settings = make_settings(model_server="fastapi")
    assert entry_script(part, settings, {}, {}) == (  # type: ignore[arg-type]
        f"export RAG_ROLE=fastapi-{part}\n{BOOTSTRAP_AND_EXEC}"
    )


@pytest.mark.parametrize("part", ["embedder", "reranker"])
def test_ovms_model_server_script_exports_the_bundle_after_app_settings(
    part: str,
) -> None:
    settings = make_settings(
        model_server="ovms", ovms_bundle="/mnt/code/ovms bundle.tar.gz"
    )
    app_settings = {"RAG_OVMS_MODELS_DIR": "/mnt/data/ovms-models"}
    assert entry_script(part, settings, {}, app_settings) == (  # type: ignore[arg-type]
        "export RAG_OVMS_MODELS_DIR=/mnt/data/ovms-models\n"
        f"export RAG_ROLE=ovms-{part}\n"
        "export RAG_OVMS_BUNDLE='/mnt/code/ovms bundle.tar.gz'\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_ovms_script_without_app_settings_has_only_computed_exports() -> None:
    settings = make_settings(model_server="ovms", ovms_bundle="/mnt/code/ovms")
    assert entry_script("reranker", settings, {}, {}) == (
        "export RAG_ROLE=ovms-reranker\n"
        "export RAG_OVMS_BUNDLE=/mnt/code/ovms\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_split_all_workflow_script_quotes_urls_and_names_the_run() -> None:
    settings = make_settings(
        topology="split-all",
        model_server="ovms",
        ovms_bundle="/mnt/code/ovms",
    )
    urls = {
        "reranker": "https://domino.test/apps/r1?a=1&b='x'",
        "embedder": "https://domino.test/apps/e1",
    }
    app_settings = {"RAG_ES_URL": "https://es.test:9200"}
    assert entry_script("workflow", settings, urls, app_settings) == (  # type: ignore[arg-type]
        "export RAG_ES_URL=https://es.test:9200\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-all\n"
        "export RAG_OVMS_RERANK_URL='https://domino.test/apps/r1?a=1&b='\"'\"'x'\"'\"''\n"
        "export RAG_OVMS_EMBEDDINGS_URL=https://domino.test/apps/e1\n"
        "export RAG_RUN_LABEL=split-all-ovms\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_split_reranker_workflow_script_has_only_the_reranker_url() -> None:
    settings = make_settings(topology="split-reranker")
    urls = {"reranker": "https://domino.test/apps/r1"}
    assert entry_script("workflow", settings, urls, {}) == (  # type: ignore[arg-type]
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_OVMS_RERANK_URL=https://domino.test/apps/r1\n"
        "export RAG_RUN_LABEL=split-reranker-fastapi\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_monolith_workflow_script_has_no_model_server_and_no_run_label() -> None:
    settings = make_settings(topology="monolith")
    assert entry_script("workflow", settings, {}, {}) == (
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_publish_creates_the_app_with_the_documented_body() -> None:
    fake = FakeDominoAppsApi("Running")
    settings = make_settings(model_server="ovms", ovms_bundle="/mnt/code/ovms")
    app = publish_app(make_client(fake), settings, "reranker", {}, {}, sleep=no_sleep)
    create = fake.calls_to("POST")[0]
    assert (create.path, create.body) == (
        APPS_PATH,
        {
            "name": "rag-reranker",
            "projectId": "proj-1",
            "visibility": "AUTHENTICATED",
            "entryPoint": "domino/app.sh",
            "version": {
                "entryScript": (
                    "export RAG_ROLE=ovms-reranker\n"
                    "export RAG_OVMS_BUNDLE=/mnt/code/ovms\n"
                    f"{BOOTSTRAP_AND_EXEC}"
                ),
                "environmentId": "env-1",
            },
        },
    )
    assert (app.name, app.status) == ("rag-reranker", "Running")


def test_create_body_sends_the_hardware_tier_only_when_set() -> None:
    fake = FakeDominoAppsApi("Running")
    settings = make_settings(topology="monolith", hardware_tier_id="tier-8")
    publish_app(make_client(fake), settings, "workflow", {}, {}, sleep=no_sleep)
    version = fake.calls_to("POST")[0].body["version"]  # type: ignore[index]
    assert {k: v for k, v in version.items() if k != "entryScript"} == {
        "environmentId": "env-1",
        "hardwareTierId": "tier-8",
    }


def test_redeploy_posts_a_new_version_instead_of_creating_an_app() -> None:
    fake = FakeDominoAppsApi("Running")
    fake.add_app("rag-reranker")
    publish_app(make_client(fake), make_settings(), "reranker", {}, {}, sleep=no_sleep)
    posts = fake.calls_to("POST")
    assert [(call.path, call.body) for call in posts] == [
        (
            f"{APPS_PATH}/app-1/versions",
            {
                "entryScript": f"export RAG_ROLE=fastapi-reranker\n{BOOTSTRAP_AND_EXEC}",
                "environmentId": "env-1",
            },
        )
    ]


def test_an_app_whose_name_only_starts_with_the_name_is_not_redeployed() -> None:
    fake = FakeDominoAppsApi("Running")
    fake.add_app("rag-reranker-old")
    app = publish_app(
        make_client(fake), make_settings(), "reranker", {}, {}, sleep=no_sleep
    )
    assert [call.path for call in fake.calls_to("POST")] == [APPS_PATH]
    assert app.name == "rag-reranker"


def test_redeploy_waits_until_the_new_version_runs() -> None:
    fake = FakeDominoAppsApi("Running")
    fake.add_app("rag-reranker")
    fake.stale_gets = 2  # the previous version still reported as Running
    app = publish_app(
        make_client(fake), make_settings(), "reranker", {}, {}, sleep=no_sleep
    )
    assert (app.version_id, len(fake.calls_to("GET")) - 1) == ("app-1-v2", 3)


def test_publish_polls_until_running_and_prints_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeDominoAppsApi("Queued", "Building", "Running")
    sleeps: list[float] = []
    publish_app(
        make_client(fake), make_settings(), "embedder", {}, {}, sleep=sleeps.append
    )
    assert sleeps == [5.0, 5.0, 5.0]
    assert capsys.readouterr().out == (
        "rag-embedder: Running https://domino.test/apps/app-1\n"
    )


@pytest.mark.parametrize("status", ["Failed", "Error", "Stopped"])
def test_a_failed_instance_names_the_app_status_and_url(status: str) -> None:
    fake = FakeDominoAppsApi("Pending", status)
    with pytest.raises(DominoDeployError) as caught:
        publish_app(
            make_client(fake), make_settings(), "embedder", {}, {}, sleep=no_sleep
        )
    assert str(caught.value) == (
        f"App rag-embedder is {status}; see its logs at https://domino.test/apps/app-1"
    )


def test_an_instance_that_appears_after_a_few_polls_is_waited_for() -> None:
    fake = FakeDominoAppsApi(None, None, "Pending", "Running")
    app = publish_app(
        make_client(fake), make_settings(), "embedder", {}, {}, sleep=no_sleep
    )
    assert app.status == "Running"


def test_an_app_that_never_gets_an_instance_says_to_start_it_once() -> None:
    fake = FakeDominoAppsApi(None)
    sleeps: list[float] = []
    with pytest.raises(DominoDeployError) as caught:
        publish_app(
            make_client(fake), make_settings(), "embedder", {}, {}, sleep=sleeps.append
        )
    assert sum(sleeps) == NO_INSTANCE_GRACE_S  # a grace, not the full ten minutes
    assert str(caught.value) == (
        "Domino created App rag-embedder but did not start it: start it once from "
        "the Apps page (https://domino.test/apps/app-1); the Domino 6.2 Apps API has "
        "no start call"
    )


def test_waiting_times_out_with_the_last_status_seen() -> None:
    fake = FakeDominoAppsApi("Pulling")
    sleeps: list[float] = []
    with pytest.raises(DominoDeployError) as caught:
        publish_app(
            make_client(fake), make_settings(), "embedder", {}, {}, sleep=sleeps.append
        )
    assert str(caught.value) == (
        "App rag-embedder is not Running after 600 s; last status: Pulling"
    )
    assert sum(sleeps) == WAIT_TIMEOUT_S


def test_deploy_all_for_split_all_creates_three_apps_in_order() -> None:
    fake = FakeDominoAppsApi("Running")
    deploy(
        make_client(fake),
        make_settings(topology="split-all"),
        "all",
        {},
        sleep=no_sleep,
    )
    creates = [call.body for call in fake.calls_to("POST")]
    assert [body["name"] for body in creates] == [  # type: ignore[index]
        "rag-embedder",
        "rag-reranker",
        "rag-workflow",
    ]
    assert creates[2]["version"]["entryScript"] == (  # type: ignore[index]
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-all\n"
        "export RAG_OVMS_RERANK_URL=https://domino.test/apps/app-2\n"
        "export RAG_OVMS_EMBEDDINGS_URL=https://domino.test/apps/app-1\n"
        "export RAG_RUN_LABEL=split-all-fastapi\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_deploying_the_workflow_without_its_reranker_names_the_make_target() -> None:
    fake = FakeDominoAppsApi("Running")
    fake.add_app("rag-embedder")
    with pytest.raises(DominoDeployError) as caught:
        deploy(make_client(fake), make_settings(), "workflow", {}, sleep=no_sleep)
    assert str(caught.value) == (
        "App rag-reranker is not deployed in project proj-1: run make deploy-reranker"
    )
    assert fake.calls_to("POST") == []


def test_deploying_an_app_the_topology_does_not_need_is_refused() -> None:
    fake = FakeDominoAppsApi("Running")
    with pytest.raises(DominoDeployError) as caught:
        deploy(make_client(fake), make_settings(topology="monolith"), "reranker", {})
    assert str(caught.value) == (
        "the monolith topology does not use App rag-reranker; it needs: rag-workflow"
    )
    assert fake.calls == []


@pytest.fixture
def domino_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Workspace's environment, run from an empty directory (no deploy.env)."""
    monkeypatch.chdir(tmp_path)
    for name in ("RAG_DEPLOY_TOPOLOGY", "RAG_DEPLOY_PROJECT_ID", "RAG_DEPLOY_ES_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOMINO_API_PROXY", PROXY)
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-1")
    monkeypatch.setenv("RAG_DEPLOY_ENVIRONMENT_ID", "env-1")


RunMain = Callable[..., tuple[int, str, str]]


@pytest.fixture
def run_main(capsys: pytest.CaptureFixture[str], domino_env: None) -> RunMain:
    """Runs ``main`` over a transport; returns (exit code, stdout, stderr)."""

    def run(
        argv: list[str], handler: Callable[[httpx.Request], httpx.Response]
    ) -> tuple[int, str, str]:
        code = 0
        try:
            main(argv, transport=httpx.MockTransport(handler), sleep=no_sleep)
        except SystemExit as exc:
            code = int(exc.code or 0)
        out, err = capsys.readouterr()
        return code, out, err

    return run


@pytest.mark.parametrize("variable", ["DOMINO_API_PROXY", "DOMINO_PROJECT_ID"])
def test_outside_a_workspace_the_missing_variable_is_named(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.delenv(variable)
    code, _, err = run_main(["status"], FakeDominoAppsApi())
    assert code == 2
    assert f"{variable} is not set" in err
    assert "runs in a Domino Workspace" in err


def test_a_403_shows_the_errors_text(run_main: RunMain) -> None:
    fake = FakeDominoAppsApi()
    fake.error = httpx.Response(
        403, json={"errors": ["User may not read project proj-1"], "requestId": "r-9"}
    )
    code, _, err = run_main(["deploy", "all"], fake)
    assert code == 2
    assert err.endswith(
        "rag-deploy: error: Domino Apps API GET /api/apps/beta/apps returned HTTP "
        "403: User may not read project proj-1\n"
    )


def test_other_failures_show_the_status_and_the_start_of_the_body(
    run_main: RunMain,
) -> None:
    fake = FakeDominoAppsApi()
    fake.error = httpx.Response(500, text="x" * 300)
    code, _, err = run_main(["deploy", "all"], fake)
    assert code == 2
    assert err.endswith(f"returned HTTP 500: {'x' * 200}\n")


READY_BODY = {
    "status": "ready",
    "checks": [
        {"name": "embedder", "ok": True, "detail": "ovms:bge-small-en-v1.5"},
        {"name": "vector_store", "ok": False, "detail": "ConnectError"},
    ],
}


def with_workflow_readyz(
    fake: FakeDominoAppsApi, token: httpx.Response, seen: list[httpx.Request]
) -> Callable[[httpx.Request], httpx.Response]:
    """The fake Apps API plus the run token endpoint and the workflow App's /readyz."""

    def handle(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://localhost:8899/access-token":
            return token
        if request.url.path.endswith("/readyz"):
            seen.append(request)
            headers = {
                "X-RAG-Topology": "split-reranker",
                "X-RAG-Run-Label": "split-reranker-fastapi",
            }
            return httpx.Response(503, json=READY_BODY, headers=headers)
        return fake(request)

    return handle


def test_status_lists_the_apps_then_the_workflow_readiness(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "split-reranker")
    fake = FakeDominoAppsApi("Running")
    fake.add_app("rag-workflow")
    fake.apps["app-1"]["currentVersion"] = {
        "id": "v1",
        "currentInstance": _instance("Running"),
    }
    seen: list[httpx.Request] = []
    handler = with_workflow_readyz(fake, httpx.Response(200, text="tok-1\n"), seen)
    code, out, _ = run_main(["status"], handler)
    assert code == 0
    assert out == (
        "rag-reranker   not deployed\n"
        "rag-workflow   Running        https://domino.test/apps/app-1\n"
        "readyz: HTTP 503, X-RAG-Topology: split-reranker, "
        "X-RAG-Run-Label: split-reranker-fastapi\n"
        "  embedder: ok ovms:bge-small-en-v1.5\n"
        "  vector_store: FAIL ConnectError\n"
    )
    assert str(seen[0].url) == "https://domino.test/apps/app-1/readyz"
    assert seen[0].headers["Authorization"] == "Bearer tok-1"


def test_status_still_succeeds_when_the_run_token_is_unavailable(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    fake = FakeDominoAppsApi("Running")
    fake.add_app("rag-workflow")
    fake.apps["app-1"]["currentVersion"] = {
        "id": "v1",
        "currentInstance": _instance("Running"),
    }
    seen: list[httpx.Request] = []
    handler = with_workflow_readyz(fake, httpx.Response(500, text="no token"), seen)
    code, out, _ = run_main(["status"], handler)
    assert code == 0
    assert out == (
        "rag-workflow   Running        https://domino.test/apps/app-1\n"
        "readyz: not checked (run token from http://localhost:8899/access-token: "
        "HTTP 500)\n"
    )
    assert seen == []


def test_status_skips_readiness_when_the_workflow_app_is_not_running(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    fake = FakeDominoAppsApi()
    fake.add_app("rag-workflow")  # no instance
    code, out, _ = run_main(["status"], fake)
    assert (code, out) == (
        0,
        "rag-workflow   no instance    https://domino.test/apps/app-1\n",
    )


def test_url_prints_only_the_app_url(run_main: RunMain) -> None:
    fake = FakeDominoAppsApi()
    fake.add_app("rag-workflow")
    fake.apps["app-1"]["url"] = "https://domino.test/apps/app-1/"
    assert run_main(["url", "workflow"], fake) == (
        0,
        "https://domino.test/apps/app-1\n",
        "",
    )


def test_url_of_an_app_not_deployed_is_an_error(run_main: RunMain) -> None:
    code, out, err = run_main(["url", "reranker"], FakeDominoAppsApi())
    assert (code, out) == (2, "")
    assert err.endswith(
        "rag-deploy: error: App rag-reranker is not deployed in project proj-1: "
        "run make deploy-reranker\n"
    )


def test_deploy_requires_the_environment_id(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_DEPLOY_ENVIRONMENT_ID")
    fake = FakeDominoAppsApi("Running")
    code, _, err = run_main(["deploy", "all"], fake)
    assert code == 2
    assert err.endswith(
        "rag-deploy: error: RAG_DEPLOY_ENVIRONMENT_ID is not set: it is the id of "
        "the Domino compute environment the Apps run in\n"
    )
    assert fake.calls == []


def test_ovms_deploy_checks_the_bundle_exists_before_publishing(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_MODEL_SERVER", "ovms")
    monkeypatch.setenv("RAG_DEPLOY_OVMS_BUNDLE", str(tmp_path / "missing.tar.gz"))
    fake = FakeDominoAppsApi("Running")
    code, _, err = run_main(["deploy", "all"], fake)
    assert code == 2
    assert err.endswith(
        f"rag-deploy: error: RAG_DEPLOY_OVMS_BUNDLE={tmp_path / 'missing.tar.gz'} "
        "does not exist: set it to the OVMS bundle folder (bin/ovms, lib/) or its "
        ".tar.gz, at a path the Apps can read\n"
    )
    assert fake.calls == []


def test_ovms_deploy_publishes_when_the_bundle_exists(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_MODEL_SERVER", "ovms")
    monkeypatch.setenv("RAG_DEPLOY_OVMS_BUNDLE", str(tmp_path))
    fake = FakeDominoAppsApi("Running")
    code, _, _ = run_main(["deploy", "reranker"], fake)
    assert code == 0
    assert [call.body["name"] for call in fake.calls_to("POST")] == [  # type: ignore[index]
        "rag-reranker"
    ]


DEPLOY_ENV = """\
RAG_DEPLOY_TOPOLOGY=split-reranker
RAG_RERANKER_MODEL=/mnt/data/models/bge reranker
RAG_ES_URL=https://es.test:9200
PATH=/should/not/pass
OPENAI_BASE_URL=https://gateway.test
RAG_EMBEDDER_MODEL=/mnt/data/models/bge-small
"""


def test_app_settings_are_the_rag_lines_of_deploy_env_sorted(tmp_path: Path) -> None:
    deploy_env = tmp_path / "deploy.env"
    deploy_env.write_text(DEPLOY_ENV)
    assert list(load_app_settings(str(deploy_env)).items()) == [
        ("RAG_EMBEDDER_MODEL", "/mnt/data/models/bge-small"),
        ("RAG_ES_URL", "https://es.test:9200"),
        ("RAG_RERANKER_MODEL", "/mnt/data/models/bge reranker"),
    ]


def test_app_settings_come_from_the_file_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_ES_URL", "https://from-the-shell.test")
    assert load_app_settings(str(tmp_path / "deploy.env")) == {}


def test_es_api_key_in_deploy_env_is_refused(tmp_path: Path) -> None:
    deploy_env = tmp_path / "deploy.env"
    deploy_env.write_text("RAG_ES_API_KEY=secret\n")
    with pytest.raises(DominoDeployError) as caught:
        load_app_settings(str(deploy_env))
    assert str(caught.value) == (
        f"RAG_ES_API_KEY is set in {deploy_env}, whose values are written into the "
        "Apps' launch scripts: set it as a Domino project or user environment "
        "variable instead"
    )


def test_a_bearer_token_in_deploy_env_is_refused(tmp_path: Path) -> None:
    deploy_env = tmp_path / "deploy.env"
    deploy_env.write_text("RAG_OVMS_AUTH=dom_pat_abc123\n")
    with pytest.raises(DominoDeployError) as caught:
        load_app_settings(str(deploy_env))
    assert str(caught.value) == (
        f"RAG_OVMS_AUTH is set in {deploy_env}, whose values are written into the "
        "Apps' launch scripts: set it as a Domino project or user environment "
        "variable instead"
    )


@pytest.mark.parametrize("auth", ["domino", ""])
def test_a_non_secret_ovms_auth_passes_through(tmp_path: Path, auth: str) -> None:
    deploy_env = tmp_path / "deploy.env"
    deploy_env.write_text(f"RAG_OVMS_AUTH={auth}\n")
    assert load_app_settings(str(deploy_env)) == {"RAG_OVMS_AUTH": auth}


def test_app_settings_come_first_so_computed_exports_win() -> None:
    settings = make_settings(topology="monolith")
    app_settings = {"RAG_ES_URL": "https://es.test/?a&b", "RAG_TOPOLOGY": "split-all"}
    assert entry_script("workflow", settings, {}, app_settings) == (
        "export RAG_ES_URL='https://es.test/?a&b'\n"
        "export RAG_TOPOLOGY=split-all\n"
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_a_run_label_in_deploy_env_replaces_the_computed_one() -> None:
    settings = make_settings(topology="split-reranker")
    urls = {"reranker": "https://domino.test/apps/r1"}
    app_settings = {"RAG_RUN_LABEL": "split-reranker-8cores"}
    assert entry_script("workflow", settings, urls, app_settings) == (  # type: ignore[arg-type]
        "export RAG_RUN_LABEL=split-reranker-8cores\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_OVMS_RERANK_URL=https://domino.test/apps/r1\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_deploy_passes_deploy_env_to_every_app(run_main: RunMain) -> None:
    Path("deploy.env").write_text(DEPLOY_ENV)  # run_main runs in an empty tmp dir
    fake = FakeDominoAppsApi("Running")
    code, _, err = run_main(["deploy", "all"], fake)
    assert code == 0, err
    scripts = [call.body["version"]["entryScript"] for call in fake.calls_to("POST")]  # type: ignore[index]
    assert len(scripts) == 2  # split-reranker, from RAG_DEPLOY_TOPOLOGY in the file
    for script in scripts:
        assert script.startswith(
            "export RAG_EMBEDDER_MODEL=/mnt/data/models/bge-small\n"
            "export RAG_ES_URL=https://es.test:9200\n"
            "export RAG_RERANKER_MODEL='/mnt/data/models/bge reranker'\n"
            "export RAG_ROLE="
        )


def test_deploy_refuses_an_es_api_key_in_deploy_env(run_main: RunMain) -> None:
    Path("deploy.env").write_text("RAG_ES_API_KEY=secret\n")
    fake = FakeDominoAppsApi("Running")
    code, _, err = run_main(["deploy", "all"], fake)
    assert code == 2 and "RAG_ES_API_KEY is set in deploy.env" in err
    assert fake.calls_to("POST") == []
