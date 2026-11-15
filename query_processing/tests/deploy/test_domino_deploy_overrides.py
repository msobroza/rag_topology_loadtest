"""The overrides: this site's values of the workflow file, in its shape, for every App.

Through ``deploy`` and ``main`` over the fake Domino, against the tests'
workflow file: each override reaches every App's launch script under the
variable of its path, after the App's free list and before qp-deploy's own
exports; a workflow setting in a free list, a misspelt key, a secret, a
placeholder or a value the workflow settings refuse stops the deploy before
anything is sent to Domino. That the workflow app reads each override back to
the same value, through ``domino/app.sh``, is in
test_entry_scripts_through_app_sh.py. The fixtures (run_main) are in
domino_deploy_fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from domino_deploy_helpers import (
    ENV_1,
    LAUNCH,
    RunMain,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_settings import (
    DeploySettings,
    OverrideValue,
    override_text,
    override_values,
)
from query_processing.deploy.publish import deploy
from rag_graphs import FIXTURE_WORKFLOW

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main
# A site's values, one of each kind: a number, a nested one, a text and a list.
SITE = {
    "http": {"timeout_s": 45, "retries": 0},
    "components": {
        "vector_store": {"elasticsearch": {"url": "https://es.site:9200"}},
        "llm": {"http": {"timeout_s": 90.5}},
    },
    "graph": {"edges": [["START", "embed", "retrieve", "rerank"], ["generate", "END"]]},
}
SITE_EXPORTS = (
    "export QP_HTTP__TIMEOUT_S=45\n"
    "export QP_HTTP__RETRIES=0\n"
    "export QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL=https://es.site:9200\n"
    "export QP_COMPONENTS__LLM__HTTP__TIMEOUT_S=90.5\n"
    "export QP_GRAPH__EDGES='"
    '[["START", "embed", "retrieve", "rerank"], ["generate", "END"]]'
    "'\n"
)


def run_deploy(apps: FakeDominoApps, settings: DeploySettings) -> None:
    deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)


def scripts(apps: FakeDominoApps) -> dict[str, str]:
    return {body.name: str(body.version.entryScript) for body in apps.creates}


def exported(script: str, variable: str) -> str | None:
    """What ``variable`` holds once the script's exports have run: the last one."""
    values = re.findall(rf"^export {variable}=(.*)$", script, re.M)
    return values[-1] if values else None


# --- how an override is written -----------------------------------------------------


def test_the_overrides_are_read_in_the_workflow_files_shape() -> None:
    overrides = yaml.safe_load(
        "http: {timeout_s: 30, retries: 2}\n"
        "components:\n"
        "  llm: {http: {timeout_s: 120.5}}\n"
        "domino: {tracing: true}\n"
        "graph: {edges: [[START, retrieve, END]]}\n"
    )
    assert override_values(make_settings(overrides=overrides).overrides) == [
        OverrideValue(("http", "timeout_s"), "30"),
        OverrideValue(("http", "retries"), "2"),
        OverrideValue(("components", "llm", "http", "timeout_s"), "120.5"),
        OverrideValue(("domino", "tracing"), "true"),  # as the shell spells it
        OverrideValue(("graph", "edges"), '[["START", "retrieve", "END"]]'),
    ]
    assert OverrideValue(("http", "retries"), "2").place == "http.retries"


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("a b", "a b"),
        (3, "3"),
        (0.2, "0.2"),
        (False, "false"),
        (None, ""),
        ([1], "[1]"),
    ],
)
def test_an_override_is_written_as_the_settings_read_it_back(
    value: object, text: str
) -> None:
    assert override_text(value) == text


# --- every App gets them ------------------------------------------------------------


def test_each_override_reaches_every_app_after_its_free_list() -> None:
    free_list = {"settings": {"QP_OVMS_MODELS_DIR": "/d/ovms"}}
    settings = make_settings(apps={"reranker": free_list}, overrides=SITE)
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, settings)
    published = scripts(apps)
    assert list(published) == ["rag-embedder", "rag-reranker", "rag-workflow"]
    assert published["rag-embedder"].startswith(SITE_EXPORTS)
    assert published["rag-reranker"].startswith(
        f"export QP_OVMS_MODELS_DIR=/d/ovms\n{SITE_EXPORTS}"
    )
    assert published["rag-workflow"].startswith(
        f"{SITE_EXPORTS}export QP_ROLE=workflow\n"
    )


def test_qp_deploys_own_exports_win_over_an_override() -> None:
    # The row places the reranker; an override of its backend comes first and
    # is replaced, as the README says of a component the row names.
    overrides = {
        "components": {"reranker": {"backend": "none"}},
        "topology_alias": "elsewhere",
    }
    settings = make_settings(topology="split-reranker", overrides=overrides)
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, settings)
    workflow = scripts(apps)["rag-workflow"]
    assert exported(workflow, "QP_COMPONENTS__RERANKER__BACKEND") == "served"
    assert exported(workflow, "QP_TOPOLOGY_ALIAS") == "split-reranker"


def test_an_override_of_what_qp_deploy_only_defaults_wins() -> None:
    overrides = {"run_label": "split-all-cpu16", "domino": {"tracing": False}}
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(overrides=overrides))
    workflow = scripts(apps)["rag-workflow"]
    assert exported(workflow, "QP_RUN_LABEL") == "split-all-cpu16"
    assert exported(workflow, "QP_DOMINO__TRACING") == "false"
    assert workflow.count("QP_RUN_LABEL=") == 1


def test_a_dry_run_shows_the_overrides_in_each_apps_entry_script(
    run_main: RunMain,
) -> None:
    overrides = "overrides: {components: {llm: {http: {timeout_s: 120}}}}\n"
    Path("deploy.yaml").write_text(ENV_1 + overrides)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    report = out[out.index("Dry run: ") :]
    line = "    export QP_COMPONENTS__LLM__HTTP__TIMEOUT_S=120\n"
    assert report.count(line) == 3  # the two model servers and the workflow


# --- refused before anything is published -------------------------------------------


def refusal(settings: DeploySettings) -> str:
    apps = FakeDominoApps(["Running"])
    with pytest.raises(DominoDeployError) as caught:
        run_deploy(apps, settings)
    assert apps.received == []  # not even a lookup reached Domino
    return str(caught.value)


TOP_LEVEL_KEYS = (
    "'calibration', 'components', 'domino', 'graph', 'http', 'local_models', "
    "'run_label', 'topology_alias'"
)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"htpp": {"retries": 0}},
            "overrides in deploy.yaml has no key 'htpp': an override is a value of "
            f"the workflow file, in its shape (expected one of: {TOP_LEVEL_KEYS})",
        ),
        (
            {"components": {"llm": {"http": {"timout_s": 120}}}},
            "overrides.components.llm.http in deploy.yaml has no key 'timout_s': an "
            "override is a value of the workflow file, in its shape (expected one "
            "of: 'backoff_s', 'retries', 'timeout_s')",
        ),
        (  # the LLM has no local block: no key its type lacks
            {"components": {"llm": {"local": {"threads": 2}}}},
            "overrides.components.llm in deploy.yaml has no key 'local': an "
            "override is a value of the workflow file, in its shape (expected one "
            "of: 'alias', 'api', 'backend', 'http', 'served', 'source_id', 'type')",
        ),
        (  # Domino's own run variables are no value of the workflow file
            {"domino_api_proxy": "http://proxy"},
            "overrides in deploy.yaml has no key 'domino_api_proxy': an override is "
            "a value of the workflow file, in its shape (expected one of: "
            f"{TOP_LEVEL_KEYS})",
        ),
    ],
)
def test_a_misspelt_key_is_refused_naming_it_and_its_place(
    overrides: dict[str, object], message: str
) -> None:
    assert refusal(make_settings(overrides=overrides)) == message


@pytest.mark.parametrize(
    ("overrides", "place", "name", "declared"),
    [
        (
            {"components": {"rerankr": {"source_id": "/d/r"}}},
            "overrides.components",
            "rerankr",
            "'embedder', 'reranker', 'llm', 'vector_store'",
        ),
        (
            {"graph": {"stages": {"rerankk": {"top_k": 5}}}},
            "overrides.graph.stages",
            "rerankk",
            "'embed', 'retrieve', 'rerank', 'generate'",
        ),
    ],
)
def test_a_component_or_stage_the_file_does_not_declare_is_refused(
    overrides: dict[str, object], place: str, name: str, declared: str
) -> None:
    assert refusal(make_settings(overrides=overrides)) == (
        f"{place} in deploy.yaml names {name!r}, which the workflow file "
        f"'{FIXTURE_WORKFLOW}' does not declare: an override changes a value the "
        f"file declares (expected one of: {declared})"
    )


def test_a_block_given_as_a_value_is_refused() -> None:
    settings = make_settings(overrides={"components": {"llm": {"http": 120}}})
    assert refusal(settings) == (
        "overrides.components.llm.http in deploy.yaml is a int; expected a mapping "
        "of its keys, as in the workflow file"
    )


def test_an_empty_value_is_refused() -> None:
    settings = make_settings(overrides={"http": {"timeout_s": None}})
    assert refusal(settings) == (
        "overrides.http.timeout_s in deploy.yaml has no value: give it one, or "
        "leave it out to keep the workflow file's"
    )


@pytest.mark.parametrize(
    ("overrides", "place", "variable"),
    [
        (
            {"components": {"llm": {"api": {"api_key": "sk-1"}}}},
            "components.llm.api.api_key",
            "QP_COMPONENTS__LLM__API__API_KEY",
        ),
        (
            {"components": {"vector_store": {"elasticsearch": {"api_key": "es-1"}}}},
            "components.vector_store.elasticsearch.api_key",
            "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__API_KEY",
        ),
    ],
)
def test_a_secret_is_refused_as_in_a_free_list(
    overrides: dict[str, object], place: str, variable: str
) -> None:
    assert refusal(make_settings(overrides=overrides)) == (
        f"overrides.{place} is set in deploy.yaml, whose values are written into "
        f"the Apps' launch scripts: set {variable} as a Domino project or user "
        "environment variable instead"
    )


@pytest.mark.parametrize("value", ["", "domino"])
def test_an_empty_or_domino_api_key_passes_through(value: str) -> None:
    overrides = {"components": {"reranker": {"api": {"api_key": value}}}}
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(overrides=overrides))
    workflow = scripts(apps)["rag-workflow"]
    assert exported(workflow, "QP_COMPONENTS__RERANKER__API__API_KEY") == (
        value or "''"
    )


def test_an_example_placeholder_is_refused() -> None:
    url = "https://<elasticsearch host>:9200"
    overrides = {"components": {"vector_store": {"elasticsearch": {"url": url}}}}
    assert refusal(make_settings(overrides=overrides)) == (
        "components.vector_store.elasticsearch.url in overrides of deploy.yaml "
        f"still holds the example placeholder {url!r}: replace it with a real value"
    )


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        (
            {"http": {"retries": -1}},
            "http.retries: Input should be greater than or equal to 0 (got -1)",
        ),
        (
            {"components": {"llm": {"http": {"timeout_s": "soon"}}}},
            "components.llm.http.timeout_s: Input should be a valid number, unable "
            "to parse string as a number (got 'soon')",
        ),
        (  # a check of the whole: an api backend needs its URL
            {"components": {"reranker": {"backend": "api"}}},
            "Value error, QP_COMPONENTS__RERANKER__BACKEND='api' needs "
            "QP_COMPONENTS__RERANKER__API__URL, the full route of the endpoint (for "
            "example https://api.cohere.com/v2/rerank); got ''",
        ),
    ],
)
def test_a_value_the_workflow_settings_refuse_is_refused(
    overrides: dict[str, object], detail: str
) -> None:
    assert refusal(make_settings(topology="monolith", overrides=overrides)) == (
        f"the workflow file '{FIXTURE_WORKFLOW}' with the overrides of deploy.yaml "
        f"makes settings the workflow app refuses: {detail}"
    )


# --- the free lists -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "variable", "path"),
    [
        (
            "defaults",
            "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL",
            "components.vector_store.elasticsearch.url",
        ),
        ("apps.reranker", "QP_LOCAL_MODELS__WORKERS", "local_models.workers"),
        ("apps.workflow", "QP_RUN_LABEL", "run_label"),
        (
            "apps.embedder",
            "QP_GRAPH__STAGES__RERANK__TOP_K",
            "graph.stages.rerank.top_k",
        ),
    ],
)
def test_a_workflow_setting_in_a_free_list_is_refused_naming_its_place(
    block: str, variable: str, path: str
) -> None:
    free_list = {"settings": {variable: "x"}}
    if block == "defaults":
        settings = make_settings(defaults={"environment_id": "env-1", **free_list})
    else:
        settings = make_settings(apps={block.removeprefix("apps."): free_list})
    assert refusal(settings) == (
        f"{variable} is set in deploy.yaml under {block}.settings, the variables "
        "the launch scripts read, but it is a value of the workflow file: write it "
        f"under overrides as {path}, in the workflow file's shape, and every App "
        "gets it"
    )


def test_the_variables_the_launch_scripts_read_still_pass_through() -> None:
    values = {
        "QP_OVMS_MODELS_DIR": "/d/ovms-models",
        "QP_BOOTSTRAP_EXTRAS": "domino,dev",
        "QP_WORKFLOW_FILE": str(FIXTURE_WORKFLOW),
    }
    defaults = {"environment_id": "env-1", "settings": values}
    apps = FakeDominoApps(["Running"])
    run_deploy(apps, make_settings(defaults=defaults))
    for script in scripts(apps).values():
        for variable, value in values.items():
            assert exported(script, variable) == value


# --- an alias row against the backends of each type ---------------------------------


@pytest.mark.parametrize(
    ("row", "name", "type_", "placement", "backends"),
    [
        (
            {"vector_store": "local"},
            "vector_store",
            "vector_store",
            "local",
            "'chroma', 'elasticsearch', 'fake'",
        ),
        ({"llm": "local"}, "llm", "llm", "local", "'api', 'fake', 'served'"),
        (
            {"vector_store": "served"},
            "vector_store",
            "vector_store",
            "served",
            "'chroma', 'elasticsearch', 'fake'",
        ),
    ],
)
def test_a_row_putting_a_component_on_a_backend_its_type_lacks_is_refused(
    row: dict[str, str], name: str, type_: str, placement: str, backends: str
) -> None:
    # Before, the row was accepted and the workflow App failed at start-up.
    table = {"odd": {"embedder": "local", **row}}
    settings = make_settings(topology="odd", topology_alias=table)
    assert refusal(settings) == (
        f"topology alias 'odd' in deploy.yaml puts component {name!r}, of type "
        f"{type_!r}, on {placement!r}, a backend its type does not have (its "
        f"backends: {backends}): leave {name!r} out of the row to keep the backend "
        "the workflow file gives it"
    )
