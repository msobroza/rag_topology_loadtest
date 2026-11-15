"""The topology alias: the deploy settings choose a topology by its alias.

Through ``deploy`` and ``main`` over the fake Domino, against the packaged
workflow file (embedder and reranker on local, the LLM on api): which Apps each
alias publishes and in which order, what the workflow App's launch script
carries, a fourth alias added to the table, and each refusal, before anything is
sent to Domino. The fixtures (run_main) are in domino_deploy_fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    BOOTSTRAP_AND_EXEC,
    ENV_1,
    INSTANCES_URL,
    LAUNCH,
    TOPOLOGY_ALIASES,
    RunMain,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.server import create_fake_domino
from fake_domino.state import FakeDominoApps
from rag_load_test.config.workflow_file import PACKAGED_WORKFLOW
from rag_load_test.deploy.apps_client import DominoDeployError
from rag_load_test.deploy.deploy_settings import DeploySettings
from rag_load_test.deploy.publish import deploy
from repo_paths import REPO_ROOT

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main
EMBEDDINGS = "/v3/embeddings"
RERANK = "/v3/rerank"
# The fourth alias: the embedder served, the reranker in the workflow App.
SPLIT_EMBEDDER = {"embedder": "served", "reranker": "local"}


def run_deploy(apps: FakeDominoApps, settings: DeploySettings) -> None:
    deploy(make_client(apps), settings, "all", launch_script=LAUNCH, sleep=no_sleep)


def created(apps: FakeDominoApps) -> list[str]:
    return [body.name for body in apps.creates]


def workflow_script(apps: FakeDominoApps) -> str:
    """The workflow App's entry script, without the exports every Agent ends with."""
    script = str(apps.creates[-1].version.entryScript)
    return script.removesuffix(f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}")


def served_url(app: int, route: str) -> str:
    return f"{INSTANCES_URL}/app-{app}-v1-i{route}"


def with_alias(alias: str, row: dict[str, str], **overrides: object) -> DeploySettings:
    """The test settings, deploying ``alias`` defined as ``row`` beside the three."""
    table = {**TOPOLOGY_ALIASES, alias: row}
    return make_settings(topology=alias, topology_alias=table, **overrides)


# --- the three aliases of today ----------------------------------------------------

TODAYS_APPS = {
    "monolith": ["rag-workflow"],
    "split-reranker": ["rag-reranker", "rag-workflow"],
    "split-all": ["rag-embedder", "rag-reranker", "rag-workflow"],
}
WORKFLOW_EXPORTS = {
    "monolith": (
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=local\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=local\n"
        "export RAG_RUN_LABEL=monolith\n"
    ),
    "split-reranker": (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-reranker\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=local\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=served\n"
        f"export RAG_COMPONENTS__RERANKER__SERVED__URL={served_url(1, RERANK)}\n"
        "export RAG_RUN_LABEL=split-reranker+reranker=fastapi\n"
    ),
    "split-all": (
        "export RAG_ROLE=workflow\n"
        "export RAG_TOPOLOGY=split-all\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=served\n"
        f"export RAG_COMPONENTS__EMBEDDER__SERVED__URL={served_url(1, EMBEDDINGS)}\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=served\n"
        f"export RAG_COMPONENTS__RERANKER__SERVED__URL={served_url(2, RERANK)}\n"
        "export RAG_RUN_LABEL=split-all+embedder=fastapi+reranker=fastapi\n"
    ),
}


@pytest.mark.parametrize(("alias", "names"), list(TODAYS_APPS.items()))
def test_each_alias_of_today_publishes_todays_apps_in_todays_order(
    alias: str, names: list[str]
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(topology=alias))
    assert created(apps) == names


@pytest.mark.parametrize("alias", list(WORKFLOW_EXPORTS))
def test_the_workflow_script_carries_each_backend_each_url_the_alias_and_the_label(
    alias: str,
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(topology=alias))
    assert workflow_script(apps) == WORKFLOW_EXPORTS[alias]


def test_the_run_label_names_each_served_component_and_its_server_kind(
    tmp_path: Path,
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(model_server="ovms", ovms_bundle=str(tmp_path)))
    label = "split-all+embedder=ovms+reranker=ovms"
    assert workflow_script(apps).endswith(f"export RAG_RUN_LABEL={label}\n")


# --- an alias added to the table ---------------------------------------------------


def test_a_fourth_alias_in_the_table_deploys_with_no_change_in_code() -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, with_alias("split-embedder", SPLIT_EMBEDDER))
    assert created(apps) == ["rag-embedder", "rag-workflow"]
    # The workflow app's own rule knows three names only: it is told monolith,
    # whose rule moves nothing, and every backend is written.
    assert workflow_script(apps) == (
        "export RAG_ROLE=monolith\n"
        "export RAG_TOPOLOGY=monolith\n"
        "export RAG_COMPONENTS__EMBEDDER__BACKEND=served\n"
        f"export RAG_COMPONENTS__EMBEDDER__SERVED__URL={served_url(1, EMBEDDINGS)}\n"
        "export RAG_COMPONENTS__RERANKER__BACKEND=local\n"
        "export RAG_RUN_LABEL=split-embedder+embedder=fastapi\n"
    )


@pytest.mark.parametrize(
    ("alias", "row", "told", "names"),
    [
        # The workflow file's local embedder stays local under split-reranker.
        ("split-reranker", {"reranker": "served"}, "split-reranker", ["rag-reranker"]),
        # split-all's rule would move the file's local reranker: monolith instead.
        ("split-all", {"embedder": "served"}, "monolith", ["rag-embedder"]),
    ],
)
def test_a_component_the_alias_does_not_name_keeps_the_workflow_files_backend(
    alias: str, row: dict[str, str], told: str, names: list[str]
) -> None:
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, with_alias(alias, row))
    assert created(apps) == [*names, "rag-workflow"]
    script = workflow_script(apps)
    named = set(re.findall(r"RAG_COMPONENTS__(\w+?)__BACKEND", script))
    assert named == {name.upper() for name in row}
    assert f"export RAG_TOPOLOGY={told}\n" in script


def test_the_workflow_file_the_workflow_app_names_is_the_one_read(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "two-rerankers.yaml"
    workflow.write_text(
        "components:\n"
        "  embedder: {backend: local, source_id: org/e}\n"
        "  reranker_fr: {type: reranker, backend: local, source_id: org/r}\n"
    )
    own = {"settings": {"RAG_WORKFLOW_FILE": str(workflow)}}
    # The packaged file, which RAG_WORKFLOW_FILE names here, has no reranker_fr.
    settings = with_alias("fr", {"reranker_fr": "local"}, apps={"workflow": own})
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, settings)
    script = workflow_script(apps)
    assert "export RAG_COMPONENTS__RERANKER_FR__BACKEND=local\n" in script


# --- refused before anything is published ------------------------------------------


def refusal(apps: FakeDominoApps, settings: DeploySettings) -> str:
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings)
    assert apps.received == []  # not even a lookup reached Domino
    return str(caught.value)


@pytest.mark.parametrize(
    ("table", "aliases"),
    [(TOPOLOGY_ALIASES, "'monolith', 'split-reranker', 'split-all'"), ({}, "none")],
)
def test_an_alias_not_in_the_table_is_refused(
    table: dict[str, dict[str, str]], aliases: str
) -> None:
    settings = make_settings(topology="split-everything", topology_alias=table)
    assert refusal(FakeDominoApps(), settings) == (
        "topology 'split-everything' is not a topology alias of deploy.yaml: set it "
        f"to a key of topology_alias (its aliases: {aliases}), or add a row for it, "
        "such as split-all: {embedder: served, reranker: served}"
    )


def test_a_row_naming_a_component_the_workflow_file_lacks_is_refused() -> None:
    settings = with_alias("typo", {"embedder": "local", "rerankr": "served"})
    assert refusal(FakeDominoApps(), settings) == (
        "topology alias 'typo' in deploy.yaml names 'rerankr', which the workflow "
        f"file '{PACKAGED_WORKFLOW}' does not declare: a row names components of "
        "that file ('embedder', 'reranker', 'llm', 'vector_store')"
    )


def test_a_served_llm_is_refused_naming_what_can_be_served() -> None:
    settings = with_alias("with-llm", {"reranker": "served", "llm": "served"})
    assert refusal(FakeDominoApps(), settings) == (
        "topology alias 'with-llm' serves component 'llm', of type 'llm', which no "
        "model server can serve: a fastapi model server serves a component of type "
        "embedder or reranker; leave 'llm' out of the row to keep the backend the "
        "workflow file gives it"
    )


def test_a_served_component_not_named_after_its_type_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("components: {reranker_fr: {type: reranker, backend: local}}")
    monkeypatch.setenv("RAG_WORKFLOW_FILE", str(workflow))
    settings = with_alias("split-fr", {"reranker_fr": "served"})
    assert refusal(FakeDominoApps(), settings) == (
        "topology alias 'split-fr' serves component 'reranker_fr', of type "
        "'reranker': a model server App serves only the component named after its "
        "type ('reranker'), the one domino/app.sh starts"
    )


def test_an_alias_that_makes_no_valid_run_label_is_refused() -> None:
    settings = with_alias("split all", {"reranker": "served"})
    assert refusal(FakeDominoApps(), settings) == (
        "topology alias 'split all' makes the run label 'split all+reranker=fastapi', "
        "which the workflow app refuses: a run label is letters, digits and ._+=:- "
        "only, 1 to 128 characters, so rename the alias"
    )


def test_an_apps_block_of_no_app_is_refused() -> None:
    settings = make_settings(apps={"orchestrator": {"kind": "agent"}})
    assert refusal(FakeDominoApps(), settings) == (
        "apps.orchestrator in deploy.yaml is no App: an apps block is keyed by "
        "workflow or by a component of the workflow file "
        f"'{PACKAGED_WORKFLOW}' ('embedder', 'reranker', 'llm', 'vector_store')"
    )


def test_a_cell_neither_local_nor_served_is_refused_before_anything_is_published(
    run_main: RunMain,
) -> None:
    Path("deploy.yaml").write_text(
        ENV_1 + "topology_alias: {split-all: {embedder: remote, reranker: served}}\n"
    )
    apps = FakeDominoApps(["Running"])
    code, out, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert (code, out) == (2, "")
    assert "topology_alias.split-all.embedder" in err
    assert "Input should be 'local' or 'served'" in err
    assert apps.received == []


# --- through the command line ------------------------------------------------------


def test_status_lists_the_apps_of_the_alias(run_main: RunMain) -> None:
    table = "topology_alias: {split-embedder: {embedder: served, reranker: local}}\n"
    Path("deploy.yaml").write_text(ENV_1 + table + "topology: split-embedder\n")
    code, out, _ = run_main(["status"], create_fake_domino(FakeDominoApps()))
    assert (code, out) == (
        0,
        "rag-embedder   not deployed\nrag-workflow   not deployed\n",
    )


EXAMPLE_YAML = REPO_ROOT / "deploy.example.yaml"


@pytest.mark.parametrize(("alias", "names"), list(TODAYS_APPS.items()))
def test_a_dry_run_of_each_alias_of_the_example_shows_its_apps_and_scripts(
    run_main: RunMain, monkeypatch: pytest.MonkeyPatch, alias: str, names: list[str]
) -> None:
    filled = re.sub(r"<([^<>]+)>", "real", EXAMPLE_YAML.read_text())
    Path("deploy.yaml").write_text(filled)
    monkeypatch.setenv("RAG_DEPLOY_TOPOLOGY", alias)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    report = out[out.index("Dry run: ") :]
    assert re.findall(r"^(rag-\w+)$", report, re.M) == names
    label = WORKFLOW_EXPORTS[alias].splitlines()[-1]
    assert f"    {label}\n" in report
