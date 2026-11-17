"""Where the prompts file is found, what it feeds, and what the workflow file may not hold."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from query_processing.config.prompts_file import PROMPTS_FILE_VAR, prompts_file
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.prompts import PromptMessage, PromptPlaceholder
from rag_graphs import FIXTURE_PROMPTS, fixture_graph, write_workflow_file

SettingsFactory = Callable[..., WorkflowSettings]


def write_prompts(path: Path, content: str) -> Path:
    """Write a prompts file of one prompt, ``said``, whose user message is ``content``."""
    prompts = {"said": [{"role": "user", "content": content}]}
    path.write_text(yaml.safe_dump(prompts), encoding="utf-8")
    return path


@pytest.fixture
def none_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty working directory, with ``QP_PROMPTS_FILE`` unset."""
    monkeypatch.delenv(PROMPTS_FILE_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_prompts_file_feeds_the_prompts_setting(
    settings_factory: SettingsFactory,
) -> None:
    prompt = settings_factory().prompts["answer_from_context"]

    assert [type(entry) for entry in prompt] == [PromptMessage, PromptMessage]
    assert prompt[0].role == "system"


def test_the_prompts_file_here_is_read_when_none_is_named(
    settings_factory: SettingsFactory, none_named: Path
) -> None:
    write_prompts(none_named / "prompts.yaml", "here")

    assert prompts_file() == Path("prompts.yaml")
    assert settings_factory().prompts["said"][0] == PromptMessage(
        role="user", content="here"
    )


def test_the_named_prompts_file_wins_over_the_working_directory(
    settings_factory: SettingsFactory,
    none_named: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_prompts(none_named / "prompts.yaml", "here")
    named = write_prompts(none_named / "other.yaml", "named")
    monkeypatch.setenv(PROMPTS_FILE_VAR, str(named))

    assert settings_factory().prompts["said"][0].content == "named"


def test_with_no_prompts_file_there_are_no_prompts(
    settings_factory: SettingsFactory, none_named: Path
) -> None:
    assert prompts_file() is None
    assert settings_factory().prompts == {}


def test_a_named_prompts_file_that_does_not_exist_is_refused(
    settings_factory: SettingsFactory,
    none_named: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROMPTS_FILE_VAR, str(none_named / "nowhere.yaml"))

    with pytest.raises(ValueError, match=r"cannot read the prompts file .*nowhere"):
        settings_factory()


def test_a_prompts_file_that_is_no_mapping_is_refused(
    settings_factory: SettingsFactory,
    none_named: Path,
) -> None:
    (none_named / "prompts.yaml").write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="holds a list; expected a mapping"):
        settings_factory()


def test_a_placeholder_entry_is_read_as_a_placeholder_and_a_stray_key_refused(
    settings_factory: SettingsFactory,
    none_named: Path,
) -> None:
    prompts = {"p": [{"placeholder": "conversational_history"}]}
    (none_named / "prompts.yaml").write_text(yaml.safe_dump(prompts), "utf-8")
    assert settings_factory().prompts["p"] == [
        PromptPlaceholder(placeholder="conversational_history")
    ]

    prompts = {"p": [{"role": "robot", "content": "x"}]}
    (none_named / "prompts.yaml").write_text(yaml.safe_dump(prompts), "utf-8")
    with pytest.raises(ValidationError, match=r"prompts\.p\.0"):
        settings_factory()


def test_a_variable_replaces_a_whole_prompt_as_json(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced = [{"role": "user", "content": "{{ conversational_query }}"}]
    monkeypatch.setenv("QP_PROMPTS__ANSWER_FROM_CONTEXT", json.dumps(replaced))

    prompt = settings_factory().prompts["answer_from_context"]

    assert prompt == [PromptMessage(role="user", content="{{ conversational_query }}")]


def test_a_prompts_key_in_the_workflow_file_is_refused(
    settings_factory: SettingsFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = write_workflow_file(tmp_path / "workflow.yaml", fixture_graph())
    blocks = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    blocks["prompts"] = yaml.safe_load(FIXTURE_PROMPTS.read_text(encoding="utf-8"))
    workflow.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    monkeypatch.setenv("QP_WORKFLOW_FILE", str(workflow))

    with pytest.raises(ValueError, match=r"has a 'prompts' key: the prompts are"):
        settings_factory()
