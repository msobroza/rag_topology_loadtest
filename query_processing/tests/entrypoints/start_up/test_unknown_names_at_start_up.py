"""Every command that opens components refuses an unknown backend, API format, key
source or orchestrator first, as a usage error, before any component opens.

The API formats and the backends are registered, so the settings refuse neither
an unknown format nor an unknown backend, and an opener would refuse it only when
its component opens: possibly after another model has loaded. Each case below
would load an in-process model if it got that far.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from query_processing.entrypoints import (
    qp_calibrate,
    qp_ingest,
    qp_model_server,
    qp_serve,
)
from rag_components import shell_settings
from sentence_transformers_fakes import FakeSentenceTransformersModule

Main = Callable[[Sequence[str] | None], int]
# Command -> its main, and the arguments that open components.
COMMANDS: dict[str, tuple[Main, list[str]]] = {
    "qp-serve": (qp_serve.main, []),
    "qp-model-server": (qp_model_server.main, ["--serve", "reranker"]),
    "qp-ingest": (qp_ingest.main, ["--synthetic", "1"]),
}
# What the environment names, and what the refusal says of it.
UNKNOWN_NAMES = {
    "an API format": (
        {"QP_COMPONENTS__LLM__API__API_FORMAT": "jina"},
        "QP_COMPONENTS__LLM__API__API_FORMAT: "
        "unknown generation API format 'jina' (expected one of: 'openai')",
    ),
    "a backend": (
        {"QP_COMPONENTS__RERANKER__BACKEND": "gpu"},
        "QP_COMPONENTS__RERANKER__BACKEND: unknown reranker backend 'gpu' "
        "(expected one of: 'api', 'fake', 'local', 'none', 'served')",
    ),
    # Every key block is read, whatever its component's backend (fake here).
    "a key source": (
        {
            "QP_COMPONENTS__LLM__API__KEY__SOURCE": "vault",
            "QP_COMPONENTS__LLM__BACKEND": "fake",
        },
        "QP_COMPONENTS__LLM__API__KEY__SOURCE: unknown key source 'vault' "
        "(expected one of: 'domino', 'env')",
    ),
    "an orchestrator": (
        {"QP_ORCHESTRATOR": "pydantic"},
        "QP_ORCHESTRATOR: unknown orchestrator 'pydantic' "
        "(expected one of: 'langgraph', 'pydantic_graph')",
    ),
}


@pytest.fixture
def fake_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FakeSentenceTransformersModule:
    """Fake models, torch; no QP_* from the shell, no ``.env``, no server run."""
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.setenv("QP_LOCAL_MODELS__RUNTIME", "torch")
    monkeypatch.chdir(tmp_path)
    for server in (qp_serve, qp_model_server):
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
def test_qp_calibrate_refuses_an_unknown_name_before_any_model_loads(
    fake_weights: FakeSentenceTransformersModule,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    unknown: str,
) -> None:
    variables, message = UNKNOWN_NAMES[unknown]
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    # qp-calibrate reports a settings mistake as one line and status 1.
    assert qp_calibrate.main(["--only", "reranker"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("qp-calibrate: cannot read the settings: ")
    assert message in line
    assert fake_weights.loaded == []
