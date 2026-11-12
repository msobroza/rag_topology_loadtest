"""The settings defaults are the one pin for the served models; the rest asks them."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.entrypoints.rag_served_models import main, served_model_assignments
from repo_paths import REPO_ROOT

ROOT = REPO_ROOT


def test_assignments_pair_each_source_id_with_its_alias(
    settings_factory: Callable[..., RagSettings],
) -> None:
    settings = settings_factory(
        embedding_model="org/embed",
        embedding_api_model="embed",
        rerank_model="org/rerank",
        rerank_api_model="rerank",
    )

    assert served_model_assignments(settings) == [
        "RERANKER_SOURCE_MODEL=org/rerank",
        "RERANKER_MODEL_NAME=rerank",
        "EMBEDDER_SOURCE_MODEL=org/embed",
        "EMBEDDER_MODEL_NAME=embed",
    ]


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No RAG_* variable from the developer's shell and no stray ``.env`` in cwd."""
    for name in [n for n in os.environ if n.startswith("RAG_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.mark.usefixtures("clean_env")
def test_rag_served_models_prints_the_settings_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "RERANKER_SOURCE_MODEL=antoinelouis/crossencoder-camembert-base-mmarcoFR",
        "RERANKER_MODEL_NAME=crossencoder-camembert-base-mmarcoFR",
        "EMBEDDER_SOURCE_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0",
        "EMBEDDER_MODEL_NAME=snowflake-arctic-embed-l-v2.0",
    ]


@pytest.mark.usefixtures("clean_env")
def test_rag_served_models_honours_overrides_and_quotes_for_eval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_RERANK_MODEL", "org/rerank")
    monkeypatch.setenv("RAG_RERANK_API_MODEL", "it's; rm -rf $HOME")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "org/embed v2")
    monkeypatch.setenv("RAG_EMBEDDING_API_MODEL", "embed")

    assert main([]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "RERANKER_SOURCE_MODEL=org/rerank",
        "RERANKER_MODEL_NAME='it'\"'\"'s; rm -rf $HOME'",
        "EMBEDDER_SOURCE_MODEL='org/embed v2'",
        "EMBEDDER_MODEL_NAME=embed",
    ]


@pytest.mark.usefixtures("clean_env")
def test_rag_served_models_output_round_trips_through_bash_eval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_RERANK_API_MODEL", "it's; echo pwned")
    main([])
    script = 'eval "$1"; printf "%s\\n" "$RERANKER_MODEL_NAME" "$EMBEDDER_SOURCE_MODEL"'

    result = subprocess.run(
        ["bash", "-c", script, "bash", capsys.readouterr().out],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "it's; echo pwned",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
    ]


def versions_env_keys() -> list[str]:
    lines = (ROOT / "ovms" / "versions.env").read_text().splitlines()
    return [line.split("=", 1)[0] for line in lines if "=" in line and line[0] != "#"]


def test_versions_env_pins_no_served_model() -> None:
    keys = versions_env_keys()
    assert "OVMS_VERSION" in keys
    assert [key for key in keys if "MODEL" in key] == []


@pytest.mark.parametrize("relative", ["domino/Dockerfile", "ovms/export_models.sh"])
def test_no_source_id_literal_outside_the_settings(relative: str) -> None:
    text = (ROOT / relative).read_text()
    assert [
        org for org in ("BAAI/", "Snowflake/", "antoinelouis/") if org in text
    ] == []
