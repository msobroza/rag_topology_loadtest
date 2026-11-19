"""``qp-ovms-config --server-flags``: the flags of the ``ovms`` server, for every App.

``--serve-llm`` prints them too, after the LLM's own. That a calibration start
takes them is tested with ``--serve`` in ``test_qp_ovms_config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from query_processing.entrypoints.qp_ovms_config import main
from rag_components import shell_settings


@pytest.fixture(autouse=True)
def _llm_from_the_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No QP_* from the shell, no ``.env``, the fixture's LLM given a source id."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QP_COMPONENTS__LLM__SOURCE_ID", "fixture-org/fixture-chat-ov")


@pytest.mark.parametrize("name", ["reranker", "embedder", "llm"])
@pytest.mark.parametrize(
    ("env", "printed"),
    [
        ({}, ""),
        (
            {"QP_OVMS_REST_WORKERS": "4", "QP_OVMS_ARGS": "--cache_dir '/a b'"},
            "--rest_workers 4 --cache_dir '/a b'",
        ),
    ],
    ids=["defaults", "named-and-passed"],
)
def test_server_flags_prints_them_as_shell_words(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    env: dict[str, str],
    printed: str,
) -> None:
    for variable, value in env.items():
        monkeypatch.setenv(variable, value)

    assert main(["--server-flags", name]) == 0

    assert capsys.readouterr().out == f"{printed}\n"


def test_an_llms_flags_come_before_the_server_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_OVMS_ARGS", "--log_level DEBUG")
    monkeypatch.setenv("QP_OVMS_MAX_NUM_SEQS", "64")

    assert main(["--serve-llm", "llm"]) == 0

    assert capsys.readouterr().out == "--max_num_seqs 64 --log_level DEBUG\n"


def test_server_flags_of_no_component_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--server-flags", "no-such-component"])

    assert exit_info.value.code == 2
    assert "embedder, reranker, llm" in capsys.readouterr().err


def test_a_flag_given_both_ways_is_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_OVMS_REST_WORKERS", "4")
    monkeypatch.setenv("QP_OVMS_ARGS", "--rest-workers=8")

    assert main(["--server-flags", "reranker"]) == 1

    [line] = capsys.readouterr().err.splitlines()
    assert line.startswith("qp-ovms-config: cannot read the settings: ")
    assert "QP_OVMS_REST_WORKERS=4" in line
    assert "--rest-workers=8" in line
