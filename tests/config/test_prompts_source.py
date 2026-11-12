"""The prompts YAML source: which file, and a clear error when it is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_load_test.config.prompts_source import (
    PACKAGED_PROMPTS,
    prompts_path,
    prompts_source,
)
from rag_load_test.config.rag_settings import RagSettings


def test_prompts_path_is_the_packaged_file_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_PROMPTS_FILE", raising=False)

    assert prompts_path() == PACKAGED_PROMPTS
    assert PACKAGED_PROMPTS.is_file()


def test_prompts_path_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAG_PROMPTS_FILE", str(tmp_path / "mine.yaml"))

    assert prompts_path() == tmp_path / "mine.yaml"


def test_prompts_source_reads_the_two_prompts(tmp_path: Path) -> None:
    path = tmp_path / "prompts.yaml"
    path.write_text("system_prompt: S\nuser_prompt: '{context} {question}'\n")

    assert prompts_source(RagSettings, path)() == {
        "system_prompt": "S",
        "user_prompt": "{context} {question}",
    }


def test_prompts_source_names_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match=r"missing\.yaml' does not exist"):
        prompts_source(RagSettings, missing)
