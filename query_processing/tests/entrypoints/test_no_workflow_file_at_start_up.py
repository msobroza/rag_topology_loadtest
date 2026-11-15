"""Every command that reads the workflow file stops, saying where it looked.

The toolkit ships no workflow file: a command finds a use case's, the file
``RAG_WORKFLOW_FILE`` names or ``workflow.yaml`` in the working directory. Run
from an empty directory with the variable unset, each one stops before anything
opens, with a message naming both places, and no traceback.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_components import shell_settings
from rag_load_test.entrypoints import (
    rag_calibrate,
    rag_export_openvino,
    rag_fetch_corpus,
    rag_ingest,
    rag_model_server,
    rag_serve,
    rag_served_models,
)

Main = Callable[[Sequence[str] | None], int]
# Each usage-error command, and the arguments it is run with.
COMMANDS: dict[str, tuple[Main, list[str]]] = {
    "rag-serve": (rag_serve.main, []),
    "rag-model-server": (rag_model_server.main, ["--serve", "reranker"]),
    "rag-ingest": (rag_ingest.main, ["--synthetic", "1"]),
    "rag-export-openvino": (rag_export_openvino.main, []),
    "rag-served-models": (rag_served_models.main, []),
    "rag-fetch-corpus": (rag_fetch_corpus.main, ["quebec-insurance", "--out", "d"]),
}


@pytest.fixture
def nowhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """An empty working directory, no RAG_* variable; the message expected."""
    for name in shell_settings(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.delenv("RAG_WORKFLOW_FILE")
    monkeypatch.chdir(tmp_path)
    for server in (rag_serve, rag_model_server):
        monkeypatch.setattr(server.uvicorn, "run", never_serve)
    return (
        "no workflow file: RAG_WORKFLOW_FILE is not set, and the working directory "
        f"{tmp_path} has no workflow.yaml; run the command from a use case's "
        "directory, or set RAG_WORKFLOW_FILE to a workflow file"
    )


def never_serve(app: object, **options: object) -> None:
    raise AssertionError(f"a server started with {options}: nothing was refused")


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_each_command_is_a_usage_error_naming_where_it_looked(
    nowhere: str, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    main, argv = COMMANDS[command]

    with pytest.raises(SystemExit) as exited:
        main(argv)

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err.endswith(f"{command}: error: {nowhere}\n")
    assert "Traceback" not in err


def test_rag_calibrate_says_where_it_looked_in_one_line(
    nowhere: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # rag-calibrate reports a settings mistake as one line and status 1.
    assert rag_calibrate.main([]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"rag-calibrate: cannot read the settings: {nowhere}\n"
