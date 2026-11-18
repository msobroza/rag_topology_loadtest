"""The README's example of a component type, "What a use case adds", stays true.

Each block of code the section shows is what a file of the test extension
holds, so that a change to the extension that the README misses fails here;
and the example's workflow file answers through the workflow app.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app
from repo_paths import REPO_ROOT, TOOLKIT_ROOT
from shout_extension.grading import BaseGrader, CapitalsGrader
from shout_extension.shouting import ShoutRequest

TESTS = TOOLKIT_ROOT / "tests"
EXAMPLE_WORKFLOW = TESTS / "helpers" / "shout_workflow.yaml"
# Every file the example shows a part of: a block is a part of one of them.
EXAMPLE_SOURCES = (
    TESTS / "helpers" / "shout_extension" / "shouting.py",
    TESTS / "helpers" / "shout_extension" / "grading.py",
    EXAMPLE_WORKFLOW,
    TESTS / "testing" / "test_component_contract.py",
)
SECTION_BEGIN, SECTION_END = (
    "### What a use case adds",
    "### What only the toolkit adds",
)
FENCED_BLOCK = re.compile(r"^```(\w+)\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def example_blocks() -> list[tuple[str, str]]:
    """The fenced blocks of the README's section, each with its language."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split(SECTION_BEGIN, 1)[1].split(SECTION_END, 1)[0]
    return FENCED_BLOCK.findall(section)


def source_holding(block: str) -> Path | None:
    """The file of the example that holds ``block`` word for word, if any."""
    for path in EXAMPLE_SOURCES:
        if block in path.read_text(encoding="utf-8"):
            return path
    return None


def test_each_block_of_the_example_is_what_the_test_extension_holds() -> None:
    code = [block for language, block in example_blocks() if language != "toml"]

    stale = [block for block in code if source_holding(block) is None]

    assert not stale, f"README blocks no file of the example holds: {stale}"
    assert {source_holding(block) for block in code} == set(EXAMPLE_SOURCES)


def test_the_example_states_the_test_runs_setting_as_the_toolkit_does() -> None:
    toml = [block for language, block in example_blocks() if language == "toml"]
    pyproject = (TOOLKIT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert len(toml) == 1
    assert set(toml[0].splitlines()) <= set(pyproject.splitlines())
    assert 'asyncio_mode = "auto"' in toml[0]


def test_the_examples_grader_answers_one_of_the_labels_it_declares() -> None:
    labels = BaseGrader.answer_labels(None)["level"]

    assert BaseGrader.backends == {"local": CapitalsGrader}
    for text in ("a", "A"):
        assert CapitalsGrader().invoke(ShoutRequest(text=text)).level in labels


@pytest.mark.parametrize(
    ("question", "answer"),
    [("how many days?", "HOW MANY DAYS?!"), ("HOW MANY DAYS?", None)],
    ids=["quiet: shouted", "loud: not"],
)
def test_the_example_workflow_answers_through_the_app(
    monkeypatch: pytest.MonkeyPatch, question: str, answer: str | None
) -> None:
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(EXAMPLE_WORKFLOW))
    settings = WorkflowSettings(_env_file=None)  # type: ignore[call-arg]

    with TestClient(create_app(CompositionRoot(settings))) as app:
        response = app.post("/query", json={"question": question})

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == answer
