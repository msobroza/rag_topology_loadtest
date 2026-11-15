"""The workflow file is the one pin for the served models; the rest asks it."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_served_models import main, served_model_lines
from rag_components import shell_settings
from repo_paths import TOOLKIT_ROOT

ROOT = TOOLKIT_ROOT


def test_each_line_names_a_component_its_type_source_id_alias_and_weight_format(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    settings = settings_factory(
        components__embedder__source_id="org/embed",
        components__embedder__alias="embed",
        components__reranker__source_id="org/rerank",
        components__reranker__alias="rerank",
        local_models__openvino_weight_format="int8",
    )

    assert served_model_lines(settings) == [
        "embedder embedder org/embed embed int8",
        "reranker reranker org/rerank rerank int8",
    ]


def test_an_embedder_or_reranker_of_any_name_is_listed_and_an_llm_is_not(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    coarse = {"type": "reranker", "backend": "fake", "source_id": "/models/coarse-rr/"}
    settings = settings_factory(components={"coarse": coarse})

    assert served_model_lines(settings) == [
        "embedder embedder fixture-org/fixture-embedder fixture-embedder fp32",
        "reranker reranker fixture-org/fixture-reranker fixture-reranker fp32",
        "coarse reranker /models/coarse-rr/ coarse-rr fp32",
    ]


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No QP_* variable from the developer's shell and no stray ``.env`` in cwd."""
    for name in shell_settings(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.mark.usefixtures("clean_env")
def test_qp_served_models_prints_the_workflow_files_components(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0

    assert capsys.readouterr().out.splitlines() == [  # fp32 is the default
        "embedder embedder fixture-org/fixture-embedder fixture-embedder fp32",
        "reranker reranker fixture-org/fixture-reranker fixture-reranker fp32",
    ]


@pytest.mark.usefixtures("clean_env")
def test_qp_served_models_honours_overrides_and_quotes_for_eval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__SOURCE_ID", "org/rerank")
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__ALIAS", "it's; rm -rf $HOME")
    monkeypatch.setenv("QP_COMPONENTS__EMBEDDER__SOURCE_ID", "org/embed v2")
    monkeypatch.setenv("QP_COMPONENTS__EMBEDDER__ALIAS", "embed")

    assert main([]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "embedder embedder 'org/embed v2' embed fp32",
        "reranker reranker org/rerank 'it'\"'\"'s; rm -rf $HOME' fp32",
    ]


@pytest.mark.usefixtures("clean_env")
def test_qp_served_models_output_round_trips_through_bash_eval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__ALIAS", "it's; echo pwned")
    main([])
    # What ovms/export_models.sh does with each line.
    script = (
        'while IFS= read -r line; do [ -n "$line" ] || continue; '
        'eval "set -- $line"; printf "%s|" "$@"; echo; done <<<"$1"'
    )

    result = subprocess.run(
        ["bash", "-c", script, "bash", capsys.readouterr().out],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "embedder|embedder|fixture-org/fixture-embedder|fixture-embedder|fp32|",
        "reranker|reranker|fixture-org/fixture-reranker|it's; echo pwned|fp32|",
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
