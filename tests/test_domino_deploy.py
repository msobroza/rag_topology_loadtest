"""Tests of ``rag_load_test.deploy`` below ``cli.main``: the topology's Apps,
entry scripts, ``deploy``, the launch script and ``deploy.env`` settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domino_deploy_helpers import (
    APPS_URL,
    BOOTSTRAP_AND_EXEC,
    DEPLOY_ENV,
    LAUNCH,
    add_app,
    make_client,
    make_settings,
    no_sleep,
)
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.fake_domino import FakeDominoApps
from rag_load_test.deploy.publish import (
    apps_for_topology,
    deploy,
    entry_script,
    launch_script_in,
)
from rag_load_test.deploy.settings import DeploySettings, load_app_settings


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


def script(part: str, settings: DeploySettings, urls: dict[str, str], app_settings: dict[str, str]) -> str:  # fmt: skip
    return entry_script(part, settings, urls, app_settings, LAUNCH)  # type: ignore[arg-type]


@pytest.mark.parametrize("part", ["embedder", "reranker"])
def test_fastapi_model_server_script_exports_its_role_and_bootstrap(part: str) -> None:
    settings = make_settings(model_server="fastapi")
    assert script(part, settings, {}, {}) == (
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
    assert script(part, settings, {}, app_settings) == (
        "export RAG_OVMS_MODELS_DIR=/mnt/data/ovms-models\n"
        f"export RAG_ROLE=ovms-{part}\n"
        "export RAG_OVMS_BUNDLE='/mnt/code/ovms bundle.tar.gz'\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_ovms_script_without_app_settings_has_only_computed_exports() -> None:
    settings = make_settings(model_server="ovms", ovms_bundle="/mnt/code/ovms")
    assert script("reranker", settings, {}, {}) == (
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
    assert script("workflow", settings, urls, app_settings) == (
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
    assert script("workflow", settings, urls, {}) == (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_OVMS_RERANK_URL=https://domino.test/apps/r1\n"
        "export RAG_RUN_LABEL=split-reranker-fastapi\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_monolith_workflow_script_has_no_model_server_and_no_run_label() -> None:
    settings = make_settings(topology="monolith")
    assert script("workflow", settings, {}, {}) == (
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_the_launch_script_is_found_in_the_repository_root(tmp_path: Path) -> None:
    (tmp_path / "domino").mkdir()
    (tmp_path / "domino" / "app.sh").write_text("#!/bin/bash\n")
    assert launch_script_in(str(tmp_path)) == f"{tmp_path}/domino/app.sh"


def test_a_directory_without_the_launch_script_says_to_run_from_the_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(DominoDeployError) as caught:
        launch_script_in(str(tmp_path))
    assert str(caught.value) == (
        f"{tmp_path} has no domino/app.sh: run rag-deploy from the repository root "
        "(for example /repos/<project>), the directory the Apps run it from"
    )


def run_deploy(apps: FakeDominoApps, settings: DeploySettings, target: str) -> None:
    deploy(make_client(apps), settings, target, {}, launch_script=LAUNCH, sleep=no_sleep)  # type: ignore[arg-type]


def test_deploy_all_for_split_all_creates_three_apps_in_order() -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(topology="split-all"), "all")
    assert [body.name for body in apps.creates] == [
        "rag-embedder",
        "rag-reranker",
        "rag-workflow",
    ]
    assert apps.creates[2].version.entryScript == (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-all\n"
        f"export RAG_OVMS_RERANK_URL={APPS_URL}/app-2\n"
        f"export RAG_OVMS_EMBEDDINGS_URL={APPS_URL}/app-1\n"
        "export RAG_RUN_LABEL=split-all-fastapi\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_deploying_the_workflow_without_its_reranker_names_the_make_target() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-embedder")
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, make_settings(), "workflow")
    assert str(caught.value) == (
        "App rag-reranker is not deployed in project proj-1: run make deploy-reranker"
    )
    assert (apps.creates, apps.new_versions) == ([], [])


def test_deploying_an_app_the_topology_does_not_need_is_refused() -> None:
    apps = FakeDominoApps(["Running"])
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, make_settings(topology="monolith"), "reranker")
    assert str(caught.value) == (
        "the monolith topology does not use App rag-reranker; it needs: rag-workflow"
    )
    assert apps.received == []


# --- deploy.env ---


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
    assert script("workflow", settings, {}, app_settings) == (
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
    assert script("workflow", settings, urls, app_settings) == (
        "export RAG_RUN_LABEL=split-reranker-8cores\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_OVMS_RERANK_URL=https://domino.test/apps/r1\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


# --- the project id ---

EXAMPLE_ENV = Path(__file__).resolve().parents[1] / "deploy.env.example"


def test_the_example_file_copied_as_is_takes_the_project_from_the_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The example has an empty RAG_DEPLOY_PROJECT_ID= line: empty means "not set".
    monkeypatch.delenv("RAG_DEPLOY_PROJECT_ID", raising=False)
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    settings = DeploySettings(_env_file=str(EXAMPLE_ENV))  # type: ignore[call-arg]
    assert settings.project_id == "proj-of-the-workspace"


def test_an_explicit_project_id_wins_over_the_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-of-the-workspace")
    monkeypatch.setenv("RAG_DEPLOY_PROJECT_ID", "proj-named-here")
    settings = DeploySettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.project_id == "proj-named-here"
