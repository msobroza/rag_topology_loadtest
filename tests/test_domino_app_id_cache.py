"""Tests of ``AppIdCache``: the App ids rag-deploy remembers, per project."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_load_test.deploy.app_id_cache import AppIdCache
from rag_load_test.deploy.apps_client import DominoDeployError


def test_a_remembered_id_is_read_back_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / ".rag-deploy-cache.json"
    AppIdCache(str(path)).remember("proj-1", "rag-workflow", "app-7")
    assert json.loads(path.read_text()) == {"proj-1": {"rag-workflow": "app-7"}}
    assert AppIdCache(str(path)).get("proj-1", "rag-workflow") == "app-7"


def test_ids_are_kept_per_project(tmp_path: Path) -> None:
    cache = AppIdCache(str(tmp_path / "cache.json"))
    cache.remember("proj-1", "rag-workflow", "app-1")
    assert cache.get("proj-2", "rag-workflow") is None


def test_a_missing_file_is_an_empty_cache_and_is_not_created_by_reading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.json"
    assert AppIdCache(str(path)).get("proj-1", "rag-workflow") is None
    assert not path.exists()


def test_a_forgotten_id_is_gone_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = AppIdCache(str(path))
    cache.remember("proj-1", "rag-workflow", "app-1")
    cache.forget("proj-1", "rag-workflow")
    assert AppIdCache(str(path)).get("proj-1", "rag-workflow") is None


def test_an_in_memory_cache_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = AppIdCache(None)
    cache.remember("proj-1", "rag-workflow", "app-1")
    assert cache.get("proj-1", "rag-workflow") == "app-1"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("text", ["{not json", '["app-1"]', '{"proj-1": {"a": 1}}'])
def test_a_file_that_cannot_be_read_names_its_path(tmp_path: Path, text: str) -> None:
    path = tmp_path / "cache.json"
    path.write_text(text)
    with pytest.raises(DominoDeployError) as caught:
        AppIdCache(str(path))
    assert str(caught.value) == (
        f"{path} is not an App id cache ({{project id: {{App name: App id}}}}): "
        "delete it and rag-deploy finds the Apps by name again"
    )
