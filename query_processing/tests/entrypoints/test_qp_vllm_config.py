"""``qp-vllm-config``: what a vLLM App passes to ``vllm serve``, and exports first.

The source id and the alias are the workflow file's component; the flags are
``VllmSettings``'. Each is one line of shell words for the launch script.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from query_processing.entrypoints.qp_vllm_config import main
from rag_components import shell_settings

EIGHT_GIB = str(8 * 2**30)


@pytest.fixture(autouse=True)
def _no_settings_from_the_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No QP_* or VLLM_* from the shell, no ``.env``."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_", "VLLM_")):
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def test_qp_vllm_config_prints_the_source_the_alias_and_the_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_VLLM_ARGS", "--hf-overrides '{\"a\": 1}'")

    assert main(["--serve", "reranker"]) == 0

    [line] = capsys.readouterr().out.splitlines()
    assert shlex.split(line) == [
        "fixture-org/fixture-reranker",
        "--served-model-name",
        "fixture-reranker",
        "--kv-cache-memory-bytes",
        EIGHT_GIB,
        "--hf-overrides",
        '{"a": 1}',
    ]


def test_the_component_is_read_by_name_from_the_workflow_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__LLM__SOURCE_ID", "fixture-org/fixture-chat")
    monkeypatch.setenv("QP_COMPONENTS__LLM__ALIAS", "chat")

    assert main(["--serve", "llm"]) == 0

    words = shlex.split(capsys.readouterr().out)
    assert words[:3] == ["fixture-org/fixture-chat", "--served-model-name", "chat"]


def test_qp_vllm_config_refuses_an_empty_source_id_in_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The fixture's LLM is served by an API: it names no weights.
    assert main(["--serve", "llm"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("qp-vllm-config: cannot serve the llm: ")
    assert "QP_COMPONENTS__LLM__SOURCE_ID" in line


@pytest.mark.parametrize("option", ["--serve", "--environment"])
def test_a_name_that_is_no_model_component_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], option: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([option, "vector_store"])

    assert exit_info.value.code == 2
    assert "embedder, reranker, llm" in capsys.readouterr().err


def test_a_flag_given_both_ways_is_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_VLLM_MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("QP_VLLM_ARGS", "--max_model_len=8192")

    assert main(["--serve", "embedder"]) == 1

    [line] = capsys.readouterr().err.splitlines()
    assert line.startswith("qp-vllm-config: cannot read the settings: ")
    assert "QP_VLLM_MAX_MODEL_LEN=4096" in line


@pytest.mark.parametrize(
    ("env", "printed"),
    [
        ({}, ""),
        (
            {"QP_VLLM_CPU_OMP_THREADS_BIND": "0-7", "QP_VLLM_CPU_RESERVED_CPUS": "2"},
            "VLLM_CPU_OMP_THREADS_BIND=0-7 VLLM_CPU_NUM_OF_RESERVED_CPU=2",
        ),
    ],
    ids=["defaults", "both-set"],
)
def test_environment_prints_name_value_words(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: dict[str, str],
    printed: str,
) -> None:
    for variable, value in env.items():
        monkeypatch.setenv(variable, value)

    assert main(["--environment", "embedder"]) == 0

    assert capsys.readouterr().out == f"{printed}\n"


@pytest.mark.parametrize("option", ["--serve", "--environment"])
def test_the_legacy_cache_variable_beside_the_named_one_is_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], option: str
) -> None:
    monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", "4")

    assert main([option, "embedder"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert "VLLM_CPU_KVCACHE_SPACE=4" in line
    assert "QP_VLLM_KV_CACHE_GIB=8" in line
