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
    DEPLOY_YAML_OVERRIDES,
    INSTANCES_URL,
    LAUNCH,
    MONOLITH_EXPORTS,
    add_app,
    make_client,
    make_settings,
    model_server_exports,
    no_sleep,
)
from fake_domino.state import FakeDominoApps
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.deploy_settings import DeploySettings
from rag_load_test.deploy.publish import (
    LaunchPaths,
    deploy,
    entry_script,
    launch_paths,
)
from rag_load_test.deploy.topology_alias import resolve_topology
from repo_paths import REPO_ROOT, TOOLKIT_ROOT

# Which Apps each topology alias needs is tested in
# test_domino_deploy_topology_alias.py; that each Role the scripts export starts
# the right server, in test_entry_scripts_through_app_sh.py.


def script(part: str, settings: DeploySettings, urls: dict[str, str]) -> str:
    topology = resolve_topology(settings)
    return entry_script(settings.for_app(part), settings, topology, urls, LAUNCH)


def with_settings(part: str, values: dict[str, str], **given: object) -> DeploySettings:  # fmt: skip
    """Settings where ``rag-<part>``'s own block sets ``values``."""
    return make_settings(apps={part: {"settings": values}}, **given)


@pytest.mark.parametrize("part", ["embedder", "reranker"])
def test_ovms_model_server_script_exports_the_bundle_after_app_settings(
    part: str,
) -> None:
    own = {"settings": {"RAG_OVMS_MODELS_DIR": "/mnt/data/ovms-models"}}
    settings = make_settings(
        apps={part: {**own, "server": "ovms"}},
        ovms_bundle="/mnt/code/ovms bundle.tar.gz",
    )
    assert script(part, settings, {}) == (
        "export RAG_OVMS_MODELS_DIR=/mnt/data/ovms-models\n"
        f"{model_server_exports(part, 'ovms')}"
        "export RAG_OVMS_BUNDLE='/mnt/code/ovms bundle.tar.gz'\n"
        f"{BOOTSTRAP_AND_EXEC}"
    )


@pytest.mark.parametrize("part", ["embedder", "reranker"])
def test_fastapi_model_server_script_names_its_component_and_kind(part: str) -> None:
    settings = make_settings(ovms_bundle="/mnt/code/ovms")
    assert script(part, settings, {}) == (
        f"{model_server_exports(part, 'fastapi')}{BOOTSTRAP_AND_EXEC}"
    )


def test_split_all_workflow_script_quotes_urls_and_names_the_run() -> None:
    settings = with_settings(
        "workflow",
        {"RAG_ES_URL": "https://es.test:9200"},
        topology="split-all",
        defaults={"environment_id": "env-1", "server": "ovms"},
        ovms_bundle="/mnt/code/ovms",
    )
    urls = {  # a "/" ending an App URL is not doubled before the route
        "reranker": "https://domino.test/apps/r'1/",
        "embedder": "https://domino.test/apps/e1",
    }
    assert script("workflow", settings, urls) == (
        "export RAG_ES_URL=https://es.test:9200\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=split-all\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=served\n"
        "export RAG_COMPONENTS__EMBEDDER__SERVED__URL=https://domino.test/apps/e1/v3/embeddings\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=served\n"
        "export RAG_COMPONENTS__RERANKER__SERVED__URL='https://domino.test/apps/r'\"'\"'1/v3/rerank'\n"
        "export RAG_RUN_LABEL=split-all+embedder=ovms+reranker=ovms\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_split_reranker_workflow_script_has_only_the_reranker_url() -> None:
    settings = make_settings(topology="split-reranker")
    urls = {"reranker": "https://domino.test/apps/r1"}
    assert script("workflow", settings, urls) == (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=split-reranker\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=local\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=served\n"
        "export RAG_COMPONENTS__RERANKER__SERVED__URL=https://domino.test/apps/r1/v3/rerank\n"
        "export RAG_RUN_LABEL=split-reranker+reranker=fastapi\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_monolith_workflow_script_has_local_backends_and_no_url() -> None:
    settings = make_settings(topology="monolith")
    assert script("workflow", settings, {}) == (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=monolith\n"
        f"{MONOLITH_EXPORTS}"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_a_workflow_published_as_a_plain_app_has_no_agent_exports() -> None:
    settings = make_settings(topology="monolith", apps={"workflow": {"kind": "app"}})
    assert script("workflow", settings, {}) == (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=monolith\n"
        f"{MONOLITH_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_an_agent_keeps_the_tracing_its_overrides_set_and_its_own_extras() -> None:
    # Domino tracing is a value of the workflow file, the extras a variable of
    # the launch script: each where it belongs.
    values = {"RAG_BOOTSTRAP_EXTRAS": "domino,dev"}
    overrides = {"domino": {"tracing": False}}
    settings = with_settings(
        "workflow", values, topology="monolith", overrides=overrides
    )
    assert script("workflow", settings, {}) == (
        "export RAG_BOOTSTRAP_EXTRAS=domino,dev\n"
        "export RAG_DOMINO__TRACING=false\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=monolith\n"
        f"{MONOLITH_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_app_settings_then_overrides_come_first_so_computed_exports_win() -> None:
    values = {"RAG_ES_URL": "https://es.test/?a&b", "RAG_ROLE": "model-server"}
    overrides = {"components": {"reranker": {"backend": "none"}}}
    settings = with_settings(
        "workflow", values, topology="monolith", overrides=overrides
    )
    assert script("workflow", settings, {}) == (
        "export RAG_ES_URL='https://es.test/?a&b'\n"
        "export RAG_ROLE=model-server\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=none\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=monolith\n"
        f"{MONOLITH_EXPORTS}{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_a_run_label_in_the_overrides_replaces_the_computed_one() -> None:
    overrides = {"run_label": "split-reranker-8cores"}
    settings = make_settings(topology="split-reranker", overrides=overrides)
    urls = {"reranker": "https://domino.test/apps/r1"}
    assert script("workflow", settings, urls) == (
        "export RAG_RUN_LABEL=split-reranker-8cores\n"
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY_ALIAS=split-reranker\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=local\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=served\n"
        "export RAG_COMPONENTS__RERANKER__SERVED__URL=https://domino.test/apps/r1/v3/rerank\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


@pytest.mark.parametrize("below", ["my_use_case", "my_use_case/variants/fr"])
def test_the_launch_script_is_the_toolkits_found_going_up_from_the_use_case(
    tmp_path: Path, below: str
) -> None:
    script = tmp_path / "query_processing" / "domino" / "app.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    use_case = tmp_path / below
    assert launch_paths(str(use_case)) == LaunchPaths(str(script), str(use_case))


def test_from_beside_the_toolkit_in_this_repository_its_launch_script_is_found() -> (
    None
):
    found = launch_paths(str(REPO_ROOT / "a_use_case"))
    assert found.script == str(TOOLKIT_ROOT / "domino" / "app.sh")


def test_a_use_case_outside_the_repository_says_where_it_looked(
    tmp_path: Path,
) -> None:
    with pytest.raises(DominoDeployError) as caught:
        launch_paths(str(tmp_path))
    assert str(caught.value) == (
        f"no query_processing/domino/app.sh in {tmp_path} or any directory above "
        "it: run rag-deploy from a use case's directory in the repository (for "
        "example /repos/<project>/<use case>, beside query_processing/), where the "
        "Apps run the role's command"
    )


def test_every_entry_script_names_the_use_case_after_what_wins_over_it() -> None:
    # A free list cannot move the Apps' working directory: rag-deploy's own wins.
    own = {"RAG_USE_CASE_DIR": "/elsewhere"}
    entry = script("reranker", with_settings("reranker", own), {})
    assert entry.splitlines()[0] == "export RAG_USE_CASE_DIR=/elsewhere"
    assert entry.endswith(BOOTSTRAP_AND_EXEC)


def run_deploy(apps: FakeDominoApps, settings: DeploySettings, target: str) -> None:
    deploy(make_client(apps), settings, target, launch=LAUNCH, sleep=no_sleep)  # type: ignore[arg-type]


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
        "export RAG_TOPOLOGY_ALIAS=split-all\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=served\n"
        f"export RAG_COMPONENTS__EMBEDDER__SERVED__URL={INSTANCES_URL}/app-1-v1-i/v3/embeddings\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=served\n"
        f"export RAG_COMPONENTS__RERANKER__SERVED__URL={INSTANCES_URL}/app-2-v1-i/v3/rerank\n"
        "export RAG_RUN_LABEL=split-all+embedder=fastapi+reranker=fastapi\n"
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
        f"{DEPLOY_YAML_OVERRIDES}"
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
    settings = with_settings("embedder", {"RAG_OVMS_MODELS_DIR": "/d/<dataset>"})
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings, "all")
    assert str(caught.value).startswith(
        "RAG_OVMS_MODELS_DIR in apps.embedder.settings of deploy.yaml still holds"
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


def published_environments(apps: FakeDominoApps) -> list[tuple[str, str | None, str | None]]:  # fmt: skip
    return [
        (c.name, c.version.environmentId, c.version.environmentRevisionId)
        for c in apps.creates
    ]


def test_the_default_revision_goes_only_with_the_default_environment() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(
        defaults={"environment_id": "env-1", "environment_revision_id": "rev-1"},
        apps={"reranker": {"environment_id": "env-2"}},
    )
    run_deploy(apps, settings, "all")
    assert published_environments(apps) == [
        ("rag-embedder", "env-1", "rev-1"),
        ("rag-reranker", "env-2", None),
        ("rag-workflow", "env-1", "rev-1"),
    ]


def test_an_apps_own_revision_is_sent_in_its_version() -> None:
    apps = FakeDominoApps(["Running"])
    own = {"environment_id": "env-2", "environment_revision_id": "rev-2"}
    settings = make_settings(apps={"reranker": own, "embedder": {"environment_revision_id": "rev-3"}})  # fmt: skip
    run_deploy(apps, settings, "all")
    assert published_environments(apps) == [
        ("rag-embedder", "env-1", "rev-3"),
        ("rag-reranker", "env-2", "rev-2"),
        ("rag-workflow", "env-1", None),
    ]


def test_a_default_revision_without_the_default_environment_is_refused() -> None:
    apps = FakeDominoApps(["Running"])
    own = {"environment_id": "env-2"}
    settings = make_settings(
        topology="split-reranker",
        defaults={"environment_revision_id": "rev-1"},
        apps={"reranker": own, "workflow": own},
    )
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings, "all")
    assert str(caught.value) == (
        "defaults.environment_revision_id 'rev-1' in deploy.yaml has no "
        "defaults.environment_id beside it: a default revision applies only together "
        "with the default environment, so set defaults.environment_id to the "
        "environment it is a build of, or set the revision in the block of each App "
        "that sets its own environment_id"
    )
    assert apps.received == []


def test_an_apps_revision_without_any_environment_is_refused() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(
        topology="monolith",
        defaults={},
        apps={"workflow": {"environment_revision_id": "rev-1"}},
    )
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings, "all")
    assert str(caught.value).startswith("No environment_id for App rag-workflow: ")
    assert apps.received == []


def test_a_placeholder_in_a_revision_is_refused_before_anything_is_sent() -> None:
    apps = FakeDominoApps(["Running"])
    defaults = {"environment_id": "env-1", "environment_revision_id": "<revision id>"}
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, make_settings(defaults=defaults), "all")
    assert str(caught.value) == (
        "environment_revision_id in defaults of deploy.yaml still holds the example "
        "placeholder '<revision id>': replace it with a real value"
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
