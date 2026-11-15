"""``rag-fake-domino``: the behaviour it serves, on the run token's port."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from fake_domino.server import main


def served_by(argv: list[str]) -> tuple[FastAPI, str, int]:
    served: list[tuple[FastAPI, str, int]] = []
    main(argv, serve=lambda app, host, port: served.append((app, host, port)))
    [(app, host, port)] = served
    return app, host, port


def test_the_command_serves_the_named_behaviour_on_the_run_token_port() -> None:
    app, host, port = served_by(["--behaviour", "fails"])
    # 8899 is where `rag-deploy status` asks for the run token (run_token_url)
    assert (host, port) == ("127.0.0.1", 8899)
    assert app.state.fake_domino_apps.statuses == ["Pending", "Failed"]


def test_the_command_refuses_an_unknown_behaviour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--behaviour", "flaky"], serve=lambda app, host, port: None)
    assert exited.value.code == 2
    assert "invalid choice: 'flaky'" in capsys.readouterr().err


def test_the_command_takes_no_emulation_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--placement", "p.json"], serve=lambda app, host, port: None)
    assert "unrecognized arguments: --placement" in capsys.readouterr().err
