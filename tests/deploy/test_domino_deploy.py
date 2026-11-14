"""Tests of ``rag_load_test.deploy`` below ``rag_deploy.main``: entry scripts,
``deploy`` and the launch script. Settings: test_deploy_settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    BOOTSTRAP_AND_EXEC,
    DEPLOY_YAML,
    INSTANCES_URL,
    LAUNCH,
    add_app,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.state import FakeDominoApps
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.deploy_settings import DeploySettings
from rag_load_test.deploy.publish import (
    deploy,
    entry_script,
    launch_script_in,
)

# Which Apps a topology needs is tested in tests/domain/test_roles.py; that each
# Role the scripts export starts the right server, in
# test_entry_scripts_through_app_sh.py.


def script(part: str, settings: DeploySettings, urls: dict[str, str]) -> str:
    return entry_script(settings.for_app(part), settings, urls, LAUNCH)  # type: ignore[arg-type]


def with_settings(part: str, values: dict[str, str], **overrides: object) -> DeploySettings:  # fmt: skip
    """Settings where ``rag-<part>``'s own block sets ``values``."""
    return make_settings(apps={part: {"settings": values}}, **overrides)


@pytest.mark.parametrize("part", ["embedder", "reranker"])
def test_ovms_model_server_script_exports_the_bundle_after_app_settings(
    part: str,
) -> None:
    settings = with_settings(
        part,
        {"RAG_OVMS_MODELS_DIR": "/mnt/data/ovms-models"},
        model_server="ovms",
        ovms_bundle="/mnt/code/ovms bundle.tar.gz",
    )
    assert script(part, settings, {}) == (
        "export RAG_OVMS_MODELS_DIR=/mnt/data/ovms-models\n"
        f"export RAG_ROLE=ovms-{part}\n"
        "export RAG_OVMS_BUNDLE='/mnt/code/ovms bundle.tar.gz'\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_split_all_workflow_script_quotes_urls_and_names_the_run() -> None:
    settings = with_settings(
        "workflow",
        {"RAG_ES_URL": "https://es.test:9200"},
        topology="split-all",
        model_server="ovms",
        ovms_bundle="/mnt/code/ovms",
    )
    urls = {  # a "/" ending an App URL is not doubled before the route
        "reranker": "https://domino.test/apps/r'1/",
        "embedder": "https://domino.test/apps/e1",
    }
    assert script("workflow", settings, urls) == (
        "export RAG_ES_URL=https://es.test:9200\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-all\n"
        "export RAG_RERANK_API_URL='https://domino.test/apps/r'\"'\"'1/v3/rerank'\n"
        "export RAG_EMBEDDING_API_URL=https://domino.test/apps/e1/v3/embeddings\n"
        "export RAG_RUN_LABEL=split-all-ovms\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_split_reranker_workflow_script_has_only_the_reranker_url() -> None:
    settings = make_settings(topology="split-reranker")
    urls = {"reranker": "https://domino.test/apps/r1"}
    assert script("workflow", settings, urls) == (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_RERANK_API_URL=https://domino.test/apps/r1/v3/rerank\n"
        "export RAG_RUN_LABEL=split-reranker-fastapi\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_monolith_workflow_script_has_no_model_server_and_no_run_label() -> None:
    settings = make_settings(topology="monolith")
    assert script("workflow", settings, {}) == (
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_a_workflow_published_as_a_plain_app_has_no_agent_exports() -> None:
    settings = make_settings(topology="monolith", apps={"workflow": {"kind": "app"}})
    assert script("workflow", settings, {}) == (
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_an_agent_keeps_the_tracing_and_extras_its_settings_set() -> None:
    values = {"RAG_DOMINO_TRACING": "0", "RAG_BOOTSTRAP_EXTRAS": "domino,dev"}
    settings = with_settings("workflow", values, topology="monolith")
    assert script("workflow", settings, {}) == (
        "export RAG_BOOTSTRAP_EXTRAS=domino,dev\n"
        "export RAG_DOMINO_TRACING=0\n"
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


def test_app_settings_come_first_so_computed_exports_win() -> None:
    values = {"RAG_ES_URL": "https://es.test/?a&b", "RAG_TOPOLOGY": "split-all"}
    settings = with_settings("workflow", values, topology="monolith")
    assert script("workflow", settings, {}) == (
        "export RAG_ES_URL='https://es.test/?a&b'\n"
        "export RAG_TOPOLOGY=split-all\n"
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_a_run_label_in_the_settings_replaces_the_computed_one() -> None:
    values = {"RAG_RUN_LABEL": "split-reranker-8cores"}
    settings = with_settings("workflow", values, topology="split-reranker")
    urls = {"reranker": "https://domino.test/apps/r1"}
    assert script("workflow", settings, urls) == (
        "export RAG_RUN_LABEL=split-reranker-8cores\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_RERANK_API_URL=https://domino.test/apps/r1/v3/rerank\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
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
    deploy(make_client(apps), settings, target, launch_script=LAUNCH, sleep=no_sleep)  # type: ignore[arg-type]


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
        f"export RAG_RERANK_API_URL={INSTANCES_URL}/app-2-v1-i/v3/rerank\n"
        f"export RAG_EMBEDDING_API_URL={INSTANCES_URL}/app-1-v1-i/v3/embeddings\n"
        "export RAG_RUN_LABEL=split-all-fastapi\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
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


def test_each_app_is_published_with_its_own_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "deploy.yaml").write_text(DEPLOY_YAML)
    monkeypatch.setenv("RAG_DEPLOY_FILE", str(tmp_path / "deploy.yaml"))
    monkeypatch.delenv("RAG_DEPLOY_TOPOLOGY", raising=False)
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, DeploySettings(project_id="proj-1"), "all")
    published = [
        (c.name, c.configurationType, c.version.environmentId, c.version.hardwareTierId)
        for c in apps.creates
    ]
    assert published == [
        ("rag-reranker", "STANDARD", "env-123", "cpu-16"),
        ("rag-workflow", "AISYSTEM", "env-123", "small"),
    ]
    reranker_script = str(apps.creates[0].version.entryScript)
    assert reranker_script.startswith(
        "export RAG_ES_URL=https://es.test:9200\nexport RAG_MODEL_WORKERS=2\n"
    )


def test_a_secret_is_refused_before_anything_is_sent() -> None:
    apps = FakeDominoApps(["Running"])
    settings = with_settings("embedder", {"RAG_ES_API_KEY": "secret"})
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings, "all")
    assert str(caught.value).startswith(
        "RAG_ES_API_KEY is set in deploy.yaml under apps.embedder.settings"
    )
    assert apps.received == []


def test_a_placeholder_is_refused_before_anything_is_sent() -> None:
    apps = FakeDominoApps(["Running"])
    settings = with_settings("embedder", {"RAG_EMBEDDING_MODEL": "/d/<dataset>"})
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings, "all")
    assert str(caught.value).startswith(
        "RAG_EMBEDDING_MODEL in apps.embedder.settings of deploy.yaml still holds"
    )
    assert apps.received == []


def test_an_app_without_an_environment_is_named_before_anything_is_sent() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(
        defaults={}, apps={"embedder": {"environment_id": "env-1"}}
    )
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings, "all")
    assert str(caught.value) == (
        "No environment_id for App rag-reranker: set defaults.environment_id (or "
        "apps.reranker.environment_id) in deploy.yaml; it is the id of the Domino "
        "compute environment the App runs in"
    )
    assert apps.received == []


def test_redeploying_a_model_server_says_the_workflow_calls_the_old_instance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-reranker")
    add_app(apps, "rag-workflow", configuration_type="AISYSTEM")
    run_deploy(apps, make_settings(topology="split-reranker"), "reranker")
    assert capsys.readouterr().out == (
        f"rag-reranker: Running {INSTANCES_URL}/app-1-v2-i\n"
        "rag-workflow still calls the previous rag-reranker instance: "
        "run make deploy-workflow\n"
    )


@pytest.mark.parametrize("target", ["reranker", "all"])
def test_no_note_when_the_workflow_is_also_deployed_or_absent(
    capsys: pytest.CaptureFixture[str], target: str
) -> None:
    run_deploy(FakeDominoApps(["Running"]), make_settings(topology="split-reranker"), target)  # fmt: skip
    assert "still calls" not in capsys.readouterr().out
