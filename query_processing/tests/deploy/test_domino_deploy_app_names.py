"""The Apps' names: ``app_name_prefix`` in the deploy settings, then the App part.

Through ``deploy`` and ``main`` over the fake Domino: which Apps are created,
found and reported under the prefix, two prefixes sharing one project, and the
prefixes refused before anything is published. The fixtures (run_main) are in
domino_deploy_fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from domino_deploy_helpers import (
    APPS_URL,
    ENV_1,
    INSTANCES_URL,
    LAUNCH,
    TOPOLOGY_ALIASES,
    RunMain,
    add_app,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.server import create_fake_domino
from fake_domino.state import FakeDominoApps
from rag_load_test.deploy.app_id_cache import AppIdCache
from rag_load_test.deploy.apps_client import DominoAppsClient, DominoDeployError
from rag_load_test.deploy.deploy_settings import DeploySettings
from rag_load_test.deploy.publish import deploy

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main
QP_RAG = "app_name_prefix: qp-rag\n"


def run_deploy(
    apps: FakeDominoApps,
    settings: DeploySettings,
    target: str,
    client: DominoAppsClient | None = None,
) -> None:
    client = client or make_client(apps)
    deploy(client, settings, target, launch=LAUNCH, sleep=no_sleep)  # type: ignore[arg-type]


def qp_rag(**overrides: object) -> DeploySettings:
    """The test settings, with every App named qp-rag-<part>."""
    return make_settings(app_name_prefix="qp-rag", **overrides)


def created(apps: FakeDominoApps) -> list[str]:
    return [body.name for body in apps.creates]


def redeployed(apps: FakeDominoApps) -> list[str]:
    return [app_id for app_id, _ in apps.new_versions]


def test_with_no_prefix_set_the_apps_keep_their_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAG_DEPLOY_APP_NAME_PREFIX", raising=False)
    monkeypatch.setenv("RAG_DEPLOY_FILE", str(tmp_path / "absent.yaml"))
    settings = DeploySettings(
        project_id="proj-1",
        topology_alias=TOPOLOGY_ALIASES,  # type: ignore[arg-type]
        defaults={"environment_id": "e"},  # type: ignore[arg-type]
    )
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, settings, "all")
    assert settings.app_name_prefix == "rag"
    assert created(apps) == ["rag-embedder", "rag-reranker", "rag-workflow"]


def test_with_a_prefix_every_app_is_created_found_and_reported_under_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, qp_rag(), "all")
    assert created(apps) == ["qp-rag-embedder", "qp-rag-reranker", "qp-rag-workflow"]
    workflow_script = str(apps.creates[2].version.entryScript)
    # The workflow App is handed the URLs of the Apps found under the prefix.
    assert f"{INSTANCES_URL}/app-2-v1-i/v3/rerank\n" in workflow_script
    assert capsys.readouterr().out == (
        f"qp-rag-embedder: Running {INSTANCES_URL}/app-1-v1-i\n"
        f"qp-rag-reranker: Running {INSTANCES_URL}/app-2-v1-i\n"
        f"qp-rag-workflow: Running {INSTANCES_URL}/app-3-v1-i\n"
    )


def test_an_app_under_the_prefix_is_redeployed_not_created_again() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "qp-rag-reranker")
    run_deploy(apps, qp_rag(topology="split-reranker"), "reranker")
    assert (created(apps), redeployed(apps)) == ([], ["app-1"])


def test_the_redeploy_note_names_the_apps_under_the_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "qp-rag-reranker")
    add_app(apps, "qp-rag-workflow", configuration_type="AISYSTEM")
    run_deploy(apps, qp_rag(topology="split-reranker"), "reranker")
    assert capsys.readouterr().out == (
        f"qp-rag-reranker: Running {INSTANCES_URL}/app-1-v2-i\n"
        "qp-rag-workflow still calls the previous qp-rag-reranker instance: "
        "run make deploy-workflow\n"
    )


def test_no_redeploy_note_for_a_workflow_app_of_another_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow", configuration_type="AISYSTEM")
    run_deploy(apps, qp_rag(topology="split-reranker"), "reranker")
    assert "still calls" not in capsys.readouterr().out


def test_a_missing_model_server_is_named_under_the_prefix() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "qp-rag-embedder")
    add_app(apps, "rag-reranker")  # another prefix's: not this deploy's reranker
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, qp_rag(), "workflow")
    assert str(caught.value) == (
        "App qp-rag-reranker is not deployed in project proj-1: "
        "run make deploy-reranker"
    )


def test_an_app_the_topology_does_not_use_is_named_under_the_prefix() -> None:
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(FakeDominoApps(), qp_rag(topology="monolith"), "reranker")
    assert str(caught.value) == (
        "the monolith topology does not use App qp-rag-reranker; "
        "it needs: qp-rag-workflow"
    )


def test_an_app_without_an_environment_is_named_under_the_prefix() -> None:
    settings = qp_rag(defaults={})
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(FakeDominoApps(), settings, "all")
    assert str(caught.value).startswith(
        "No environment_id for App qp-rag-embedder: set defaults.environment_id "
        "(or apps.embedder.environment_id)"
    )


def test_two_prefixes_in_one_project_make_two_sets_of_apps() -> None:
    apps = FakeDominoApps(["Running"])
    client = make_client(apps, cache=AppIdCache(None))  # one cache for both, as on disk
    run_deploy(apps, make_settings(app_name_prefix="rag"), "all", client)
    # qp-rag-workflow contains rag-workflow: an App is found by its exact name.
    run_deploy(apps, qp_rag(), "all", client)
    assert created(apps) == [
        "rag-embedder",
        "rag-reranker",
        "rag-workflow",
        "qp-rag-embedder",
        "qp-rag-reranker",
        "qp-rag-workflow",
    ]
    # Each workflow App calls the model servers of its own set.
    second_workflow = str(apps.creates[5].version.entryScript)
    assert f"{INSTANCES_URL}/app-5-v1-i/v3/rerank" in second_workflow
    assert f"{INSTANCES_URL}/app-4-v1-i/v3/embeddings" in second_workflow
    # Deploying the first prefix again redeploys its own three Apps, nothing else.
    run_deploy(apps, make_settings(app_name_prefix="rag"), "all", client)
    assert len(apps.creates) == 6
    assert redeployed(apps) == ["app-1", "app-2", "app-3"]


# --- the prefix through the command line ---


def test_status_and_url_find_the_apps_under_the_prefix(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(ENV_1 + QP_RAG + "topology: split-reranker\n")
    apps = FakeDominoApps([None])
    add_app(apps, "rag-reranker")  # another prefix's, in the same project
    add_app(apps, "qp-rag-reranker")
    code, out, _ = run_main(["status"], create_fake_domino(apps))
    assert (code, out) == (
        0,
        f"qp-rag-reranker   no instance    {APPS_URL}/app-2\n"
        "qp-rag-workflow   not deployed\n",
    )
    code, _, err = run_main(["url", "workflow"], create_fake_domino(apps))
    assert (code, err.splitlines()[-1]) == (
        2,
        "rag-deploy: error: App qp-rag-workflow is not deployed in project proj-1: "
        "run make deploy-workflow",
    )


def test_the_prefix_can_come_from_the_environment(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DEPLOY_APP_NAME_PREFIX", "qp-rag")
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", "monolith")
    apps = FakeDominoApps(["Running"])
    code, out, _ = run_main(["deploy", "workflow"], create_fake_domino(apps))
    assert (code, created(apps)) == (0, ["qp-rag-workflow"])
    assert out.startswith("qp-rag-workflow: Running ")


# --- prefixes that would not make a valid App name ---

BAD_PREFIXES = [
    "",  # it would make "-workflow"
    "QP-rag",
    "qp_rag",
    "qp rag",
    "qp/rag",
    "-rag",
    "rag-",
    "qp--rag",
    "1rag",
]


def refusal(prefix: str) -> str:
    return (
        f"app_name_prefix {prefix!r} would not make a valid App name: it must be "
        "lowercase letters and digits in words joined by single hyphens, starting "
        "with a letter, such as 'rag' or 'qp-rag' (rag-deploy's rule, not Domino's)"
    )


@pytest.mark.parametrize("prefix", BAD_PREFIXES)
def test_a_prefix_that_would_not_make_a_valid_app_name_is_refused(prefix: str) -> None:
    with pytest.raises(ValidationError) as caught:
        make_settings(app_name_prefix=prefix)
    assert refusal(prefix) in str(caught.value)


@pytest.mark.parametrize("prefix", ["rag", "qp-rag", "uc2", "a", "qp-rag-2"])
def test_a_prefix_of_lowercase_words_is_accepted(prefix: str) -> None:
    settings = make_settings(app_name_prefix=prefix)
    assert settings.app_name("workflow") == f"{prefix}-workflow"


def test_a_refused_prefix_publishes_nothing(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(ENV_1 + "app_name_prefix: QP_RAG\n")
    apps = FakeDominoApps(["Running"])
    code, out, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert (code, out) == (2, "")
    assert refusal("QP_RAG") in err
    assert apps.received == []
