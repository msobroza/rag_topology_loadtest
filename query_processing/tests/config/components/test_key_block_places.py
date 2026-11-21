"""Where a ``key`` block comes from: a place above the workflow file replaces it whole.

The places a setting comes from are, highest first, the values a command gives,
the environment, ``.env``, then the workflow file. A key block that a place
above the file gives with a ``source`` is taken from the places above alone;
one that sets another key, and no ``source``, changes that key alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from query_processing.config.components.key_settings import KeySettings
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import FIXTURE_WORKFLOW

FILE_BLOCK = {"source": "env", "variable": "OPENAI_API_KEY"}
# The two blocks that hold a key block: a model's endpoint and the store.
KEYED = pytest.mark.parametrize(
    ("component", "block"),
    [("llm", "api"), ("vector_store", "elasticsearch")],
    ids=["llm-api", "vector-store-elasticsearch"],
)


@pytest.fixture(autouse=True)
def keyed_workflow_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The fixture's workflow file, each keyed block naming FILE_BLOCK, in tmp_path."""
    workflow = yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))
    components = workflow["components"]
    components["llm"].setdefault("api", {})["key"] = FILE_BLOCK
    components["vector_store"].setdefault("elasticsearch", {})["key"] = FILE_BLOCK
    path = tmp_path / "keyed.yaml"
    path.write_text(yaml.safe_dump(workflow), encoding="utf-8")
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))
    monkeypatch.chdir(tmp_path)


def key_block(settings: WorkflowSettings, component: str, block: str) -> KeySettings:
    return getattr(settings.components[component], block).key


def variable(component: str, block: str, *levels: str) -> str:
    return WorkflowSettings.variable_of("components", component, block, "key", *levels)


@KEYED
@pytest.mark.parametrize("place", ["environment", "dotenv", "json"])
def test_a_source_given_above_the_file_replaces_its_whole_key_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    component: str,
    block: str,
    place: str,
) -> None:
    if place == "environment":
        monkeypatch.setenv(variable(component, block, "source"), "domino")
    elif place == "json":
        monkeypatch.setenv(variable(component, block), json.dumps({"source": "domino"}))
    else:
        line = f"{variable(component, block, 'source')}=domino\n"
        (tmp_path / ".env").write_text(line, encoding="utf-8")

    settings = WorkflowSettings()

    assert key_block(settings, component, block) == KeySettings(source="domino")


@KEYED
def test_a_source_given_by_a_commands_values_replaces_the_files_block(
    component: str, block: str
) -> None:
    given = {"components": {component: {block: {"key": {"source": "domino"}}}}}

    settings = WorkflowSettings(_env_file=None, **given)

    assert key_block(settings, component, block) == KeySettings(source="domino")


@KEYED
def test_another_key_given_above_the_file_changes_that_key_alone(
    monkeypatch: pytest.MonkeyPatch, component: str, block: str
) -> None:
    monkeypatch.setenv(variable(component, block, "variable"), "OTHER_KEY")

    settings = WorkflowSettings(_env_file=None)

    expected = KeySettings(source="env", variable="OTHER_KEY")
    assert key_block(settings, component, block) == expected
