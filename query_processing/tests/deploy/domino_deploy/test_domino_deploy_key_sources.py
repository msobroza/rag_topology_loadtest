"""What the deploy tool says of API keys: no key in the deploy file, a ``key`` block may be.

Through ``deploy`` and ``main`` over the fake Domino, against the tests' workflow
file: a ``settings`` list that gives a value to a variable an ``env`` key source
names, or to any name ending in ``_API_KEY``, stops the deploy before anything is
sent to Domino, and so does an old ``api_key``, with the line start-up prints; an
override that sets a ``key`` block reaches every App's launch script. No refusal
holds the value refused. What start-up refuses of a ``key`` block is in
test_domino_deploy_start_up_refusals.py. The fixtures (run_main) are in
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
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.publish import deploy
from rag_graphs import FIXTURE_WORKFLOW

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main
SECRET = "sk-the-key-itself"
LLM_KEY = {"source": "env", "variable": "LLM_KEY"}  # a name no _API_KEY rule sees
STORE_KEY = {"source": "env", "variable": "STORE_KEY"}


def refusal(settings: DeploySettings) -> str:
    """The deploy's refusal of ``settings``, once nothing reached Domino."""
    apps = FakeDominoApps(["Running"])
    with pytest.raises(DominoDeployError) as caught:
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert apps.received == []  # not even a lookup reached Domino
    said = str(caught.value)
    assert SECRET not in said
    return said


def with_free_list(
    block: str, values: dict[str, str], **given: object
) -> DeploySettings:
    """Settings whose free list ``block`` (``defaults`` or ``apps.<part>``) is ``values``."""
    if block == "defaults":
        defaults = {"environment_id": "env-1", "settings": values}
        return make_settings(defaults=defaults, **given)
    return make_settings(
        apps={block.removeprefix("apps."): {"settings": values}}, **given
    )


def workflow_file_with_store_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tests' workflow file, its store naming STORE_KEY in its own key block."""
    data = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    data["components"]["vector_store"]["elasticsearch"]["key"] = STORE_KEY
    workflow_file = tmp_path / "workflow.yaml"
    workflow_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow_file))


# --- a variable an env key source names -----------------------------------------------


def variable_message(variable: str, block: str, place: str) -> str:
    return (
        f"{variable} is set in deploy.yaml under {block}.settings, whose values are "
        f"written into the Apps' launch scripts, but the key source of {place} "
        "reads an API key from it: set it as a Domino project or user environment "
        "variable instead"
    )


@pytest.mark.parametrize("block", ["defaults", "apps.workflow", "apps.embedder"])
def test_a_variable_an_overrides_env_source_names_is_refused_in_a_free_list(
    block: str,
) -> None:
    overrides = {"components": {"llm": {"api": {"key": LLM_KEY}}}}
    settings = with_free_list(block, {"LLM_KEY": SECRET}, overrides=overrides)
    assert refusal(settings) == variable_message(
        "LLM_KEY", block, "components.llm.api.key"
    )


def test_a_variable_the_workflow_files_env_source_names_is_refused_in_a_free_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workflow_file_with_store_key(monkeypatch, tmp_path)
    settings = with_free_list("apps.reranker", {"STORE_KEY": SECRET})
    assert refusal(settings) == variable_message(
        "STORE_KEY", "apps.reranker", "components.vector_store.elasticsearch.key"
    )


def test_a_variable_no_env_source_names_passes_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The file names STORE_KEY; LLM_KEY is named only once an override says so.
    workflow_file_with_store_key(monkeypatch, tmp_path)
    settings = with_free_list("defaults", {"LLM_KEY": "not a key of ours"})
    apps = FakeDominoApps(["Running"])
    deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert len(apps.creates) == 3


# --- a name ending in _API_KEY, whatever its value ------------------------------------


@pytest.mark.parametrize("value", ["", "domino", SECRET])
@pytest.mark.parametrize("block", ["defaults", "apps.workflow"])
def test_a_name_ending_in_api_key_is_refused_whatever_its_value(
    block: str, value: str
) -> None:
    settings = with_free_list(block, {"QP_ES_API_KEY": value})
    assert refusal(settings) == (
        f"QP_ES_API_KEY is set in deploy.yaml under {block}.settings, whose values "
        "are written into the Apps' launch scripts: set it as a Domino project or "
        "user environment variable instead"
    )


# --- a key block in the overrides ----------------------------------------------------


def exported(script: str, variable: str) -> str | None:
    values = re.findall(rf"^export {variable}=(.*)$", script, re.M)
    return values[-1] if values else None


def test_a_key_block_in_the_overrides_reaches_every_apps_launch_script() -> None:
    overrides = {
        "components": {
            "llm": {"api": {"key": LLM_KEY}},
            "reranker": {"api": {"key": {"source": "domino"}}},
        }
    }
    apps = FakeDominoApps(["Running"])
    settings = make_settings(overrides=overrides)
    deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    scripts = [str(create.version.entryScript) for create in apps.creates]
    assert len(scripts) == 3
    for script in scripts:
        assert exported(script, "QP_COMPONENTS__LLM__API__KEY__SOURCE") == "env"
        assert exported(script, "QP_COMPONENTS__LLM__API__KEY__VARIABLE") == "LLM_KEY"
        assert exported(script, "QP_COMPONENTS__RERANKER__API__KEY__SOURCE") == "domino"


def test_a_dry_run_shows_a_key_block_of_the_overrides(run_main: RunMain) -> None:
    key = "overrides: {components: {llm: {api: {key: {source: domino}}}}}\n"
    Path("deploy.yaml").write_text(ENV_1 + key)
    code, out, err = run_main(["--dry-run", "deploy", "all"], None)
    assert (code, err) == (0, "")
    line = "    export QP_COMPONENTS__LLM__API__KEY__SOURCE=domino\n"
    assert out.count(line) == 3


# --- the old api_key ------------------------------------------------------------------


def start_up_line(components: dict[str, object]) -> str:
    """What start-up says of the old ``api_key`` in ``components``."""
    with pytest.raises(ValueError, match="is no longer read") as caught:
        WorkflowSettings(_env_file=None, components=components)  # type: ignore[call-arg]
    return str(caught.value)


@pytest.mark.parametrize("value", ["", "domino", SECRET])
@pytest.mark.parametrize(
    ("name", "holder"),
    [("reranker", "api"), ("vector_store", "elasticsearch")],
    ids=["endpoint", "store"],
)
def test_an_old_api_key_override_gets_the_line_start_up_prints(
    name: str, holder: str, value: str
) -> None:
    given: dict[str, object] = {name: {holder: {"api_key": value}}}
    line = start_up_line(given)
    assert "key block" in line
    settings = make_settings(overrides={"components": given})
    assert refusal(settings) == f"overrides in deploy.yaml: {line}"


def test_an_old_api_key_in_the_workflow_file_gets_the_line_start_up_prints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    data["components"]["llm"]["api"]["api_key"] = SECRET
    workflow_file = tmp_path / "workflow.yaml"
    workflow_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow_file))
    line = start_up_line({"llm": {"api": {"api_key": SECRET}}})
    assert refusal(make_settings()) == (
        f"the workflow file {str(workflow_file)!r} with the overrides of deploy.yaml "
        f"makes settings the workflow app refuses: {line}"
    )
