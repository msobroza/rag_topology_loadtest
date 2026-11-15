"""Every command that opens components refuses an unknown backend or API format
first, as a usage error, before any component opens.

Since the API formats are registered, the settings no longer refuse an unknown
one, and an opener refuses it only when its component opens: possibly after
another model has loaded. The settings still refuse an unknown backend
themselves. Each case below would load an in-process model if it got that far.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_load_test.entrypoints import (
    rag_calibrate,
    rag_ingest,
    rag_model_server,
    rag_serve,
)
from sentence_transformers_fakes import FakeSentenceTransformersModule

Main = Callable[[Sequence[str] | None], int]
# Command -> its main, and the arguments that open components.
COMMANDS: dict[str, tuple[Main, list[str]]] = {
    "rag-serve": (rag_serve.main, []),
    "rag-model-server": (rag_model_server.main, ["--serve", "reranker"]),
    "rag-ingest": (rag_ingest.main, ["--synthetic", "1"]),
}
# What the environment names, and what the refusal says of it.
UNKNOWN_NAMES = {
    "an API format": (
        {"RAG_GENERATION_API_FORMAT": "jina"},
        "unknown generation API format 'jina' (expected one of: 'openai')",
    ),
    "a backend": (
        {"RAG_RERANK_BACKEND": "gpu"},
        "Input should be 'local', 'api', 'fake' or 'none'",
    ),
}


@pytest.fixture
def fake_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FakeSentenceTransformersModule:
    """Fake models, torch; no RAG_* from the shell, no ``.env``, no server run."""
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name in [n for n in os.environ if n.startswith(("RAG_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "torch")
    monkeypatch.chdir(tmp_path)
    for server in (rag_serve, rag_model_server):
        monkeypatch.setattr(server.uvicorn, "run", never_serve)
    return module


def never_serve(app: object, **options: object) -> None:
    raise AssertionError(f"a server started with {options}: nothing was refused")


@pytest.mark.parametrize("unknown", sorted(UNKNOWN_NAMES))
@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_an_unknown_name_is_a_usage_error_before_anything_opens(
    fake_weights: FakeSentenceTransformersModule,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    unknown: str,
) -> None:
    main, argv = COMMANDS[command]
    variables, message = UNKNOWN_NAMES[unknown]
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit) as exited:
        main(argv)

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert err.startswith("usage: ")
    assert "Traceback" not in err
    assert fake_weights.loaded == []


@pytest.mark.parametrize("unknown", sorted(UNKNOWN_NAMES))
def test_rag_calibrate_refuses_an_unknown_name_before_any_model_loads(
    fake_weights: FakeSentenceTransformersModule,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    unknown: str,
) -> None:
    variables, message = UNKNOWN_NAMES[unknown]
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    # rag-calibrate reports a settings mistake as one line and status 1.
    assert rag_calibrate.main(["--only", "reranker"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("rag-calibrate: cannot read the settings: ")
    assert message in line
    assert fake_weights.loaded == []
