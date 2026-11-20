"""The LLM served by OVMS in a model server App of its own.

A topology alias row says ``llm: served`` and the LLM's App block (or the
defaults) ``server: ovms``: ``qp-deploy`` publishes ``rag-llm`` before the
workflow App and hands the workflow App the LLM's backend and its URL on OVMS's
chat completions route. Through ``deploy`` and ``main`` over the fake Domino,
against the tests' workflow file (the LLM on api there). The fixtures (run_main)
are in domino_deploy_fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    BOOTSTRAP_AND_EXEC,
    INSTANCES_URL,
    LAUNCH,
    PREFIX_YAML,
    TOPOLOGY_ALIASES,
    WHEELS_YAML,
    RunMain,
    add_app,
    launched,
    make_client,
    make_settings,
    model_server_exports,
    no_sleep,
)
from fake_domino.server import create_fake_domino
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.publish import deploy

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main
CHAT = "/v3/chat/completions"
# Every component served, the LLM by OVMS: four Apps.
SPLIT_ALL_LLM = {"embedder": "served", "reranker": "served", "llm": "served"}
# Only the LLM served: the embedder and the reranker in the workflow App.
SPLIT_LLM = {"embedder": "local", "reranker": "local", "llm": "served"}
LLM_ON_OVMS = {"llm": {"server": "ovms"}}


def llm_settings(
    tmp_path: Path, row: dict[str, str] = SPLIT_ALL_LLM, **given: object
) -> DeploySettings:
    """The test settings deploying ``row`` as ``with-llm``, an OVMS bundle that exists."""
    values: dict[str, object] = {"apps": LLM_ON_OVMS, **given}
    values.setdefault("ovms_bundle", str(tmp_path))
    return make_settings(
        topology="with-llm",
        topology_alias={**TOPOLOGY_ALIASES, "with-llm": row},
        **values,
    )


def run_deploy(apps: FakeDominoApps, settings: DeploySettings, target: str) -> None:
    deploy(make_client(apps), settings, target, launch=LAUNCH, sleep=no_sleep)  # type: ignore[arg-type]


def created(apps: FakeDominoApps) -> list[str]:
    return [body.name for body in apps.creates]


def script_of(apps: FakeDominoApps, name: str) -> str:
    return next(
        str(body.version.entryScript) for body in apps.creates if body.name == name
    )


def served_url(app: int, route: str) -> str:
    return f"{INSTANCES_URL}/app-{app}-v1-i{route}"


# --- deploy over the fake Domino -----------------------------------------------------


def test_deploy_all_publishes_the_llm_after_the_model_servers_before_the_workflow(
    tmp_path: Path,
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, llm_settings(tmp_path), "all")
    assert created(apps) == ["rag-embedder", "rag-reranker", "rag-llm", "rag-workflow"]


@pytest.mark.parametrize(
    "kinds",
    [
        {"apps": LLM_ON_OVMS},
        {"apps": {}, "defaults": {"environment_id": "env-1", "server": "ovms"}},
    ],
    ids=["apps.llm.server", "defaults.server"],
)
def test_the_llm_app_runs_the_ovms_kind_with_the_bundle(
    tmp_path: Path, kinds: dict[str, object]
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, llm_settings(tmp_path, SPLIT_LLM, **kinds), "all")
    assert script_of(apps, "rag-llm") == (
        f"{model_server_exports('llm', 'ovms')}"
        f"export QP_OVMS_BUNDLE={tmp_path}\n{BOOTSTRAP_AND_EXEC}"
    )


def test_the_workflow_app_calls_the_llm_on_its_chat_route_and_labels_the_run(
    tmp_path: Path,
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, llm_settings(tmp_path, SPLIT_LLM), "all")
    assert script_of(apps, "rag-workflow") == (
        "export QP_ROLE=workflow\n"
        "export QP_TOPOLOGY_ALIAS=with-llm\n"
        "export QP_COMPONENTS__EMBEDDER__BACKEND=local\n"
        "export QP_COMPONENTS__RERANKER__BACKEND=local\n"
        "export QP_COMPONENTS__LLM__BACKEND=served\n"
        f"export QP_COMPONENTS__LLM__SERVED__URL={served_url(1, CHAT)}\n"
        "export QP_RUN_LABEL=with-llm+llm=ovms\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


@pytest.mark.parametrize("alias", list(TOPOLOGY_ALIASES))
def test_an_alias_that_does_not_serve_the_llm_writes_nothing_of_it(alias: str) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(topology=alias, apps=LLM_ON_OVMS), "all")
    scripts = "\n".join(str(body.version.entryScript) for body in apps.creates)
    assert "rag-llm" not in created(apps)
    assert "QP_COMPONENTS__LLM__" not in scripts


@pytest.mark.parametrize("target", ["llm", "all"])
def test_an_llm_on_ovms_without_the_bundle_is_refused_before_anything_is_sent(
    target: str,
) -> None:
    apps = FakeDominoApps(["Running"])
    with pytest.raises(DominoDeployError, match="ovms_bundle= does not exist"):
        run_deploy(apps, llm_settings(Path(), SPLIT_LLM, ovms_bundle=""), target)
    assert apps.received == []


def test_deploying_the_llm_alone_says_the_workflow_calls_the_old_instance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-llm")
    add_app(apps, "rag-workflow", configuration_type="AISYSTEM")
    run_deploy(apps, llm_settings(tmp_path, SPLIT_LLM), "llm")
    assert (apps.creates, len(apps.new_versions)) == ([], 1)
    assert capsys.readouterr().out.endswith(
        "rag-workflow still calls the previous rag-llm instance: "
        "run make deploy-workflow\n"
    )


def test_a_workflow_whose_llm_is_not_deployed_names_the_make_target(
    tmp_path: Path,
) -> None:
    apps = FakeDominoApps(["Running"])
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, llm_settings(tmp_path, SPLIT_LLM), "workflow")
    assert str(caught.value) == (
        "App rag-llm is not deployed in project proj-1: run make deploy-llm"
    )
    assert (apps.creates, apps.new_versions) == ([], [])


# --- through the command line --------------------------------------------------------

# A deploy file serving the LLM by OVMS: its App's settings, and the model OVMS
# serves and its served name, under overrides.components.llm.
LLM_DEPLOY_YAML = PREFIX_YAML + WHEELS_YAML + """\
topology_alias:
  split-llm: {embedder: local, reranker: local, llm: served}
topology: split-llm
defaults:
  environment_id: env-1
apps:
  llm:
    server: ovms
    settings: {QP_OVMS_NUM_STREAMS: 2}
overrides:
  components:
    llm: {source_id: fixture-org/fixture-chat-int4-ov, alias: fixture-chat}
"""
LLM_OVERRIDES = (
    "export QP_COMPONENTS__LLM__SOURCE_ID=fixture-org/fixture-chat-int4-ov\n"
    "export QP_COMPONENTS__LLM__ALIAS=fixture-chat\n"
)


def test_a_dry_run_prints_the_llm_app_and_the_workflow_app_naming_its_url(
    run_main: RunMain, tmp_path: Path
) -> None:
    Path("deploy.yaml").write_text(LLM_DEPLOY_YAML + f"ovms_bundle: {tmp_path}\n")

    code, out, err = run_main(["--dry-run", "deploy", "all"], None)

    assert (code, err) == (0, "")
    report = out[out.index("Dry run: ") :]
    assert re.findall(r"^(rag-\w+)$", report, re.M) == ["rag-llm", "rag-workflow"]
    scripts = re.findall(r"entryScript:\n((?:    .*\n)+)", report)
    llm, workflow = (script.replace("    ", "") for script in scripts)
    assert llm == (
        f"export QP_OVMS_NUM_STREAMS=2\n{LLM_OVERRIDES}"
        f"{model_server_exports('llm', 'ovms')}"
        f"export QP_OVMS_BUNDLE={tmp_path}\n{launched()}\n"
    )
    assert workflow.startswith(LLM_OVERRIDES + "export QP_ROLE=workflow\n")
    assert (
        "export QP_COMPONENTS__LLM__BACKEND=served\n"
        f"export QP_COMPONENTS__LLM__SERVED__URL={served_url(1, CHAT)}\n"
        "export QP_RUN_LABEL=split-llm+llm=ovms\n"
    ) in workflow


def test_status_lists_the_llm_app_before_the_workflow(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text(LLM_DEPLOY_YAML)
    code, out, _ = run_main(["status"], create_fake_domino(FakeDominoApps()))
    assert (code, out) == (
        0,
        "rag-llm        not deployed\nrag-workflow   not deployed\n",
    )
