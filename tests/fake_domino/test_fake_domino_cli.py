"""``rag-fake-domino``: the behaviour it serves and the App proxy's flags."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from fake_domino.proxy_settings import FakeDominoProxySettings, UnauthenticatedResponse
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


def test_without_flags_the_proxy_has_its_defaults() -> None:
    app, _, _ = served_by([])
    assert app.state.fake_domino_proxy.settings == FakeDominoProxySettings()
    assert FakeDominoProxySettings() == FakeDominoProxySettings(
        unauthenticated=UnauthenticatedResponse.UNAUTHORIZED,
        identity_jwt=True,
        keep_prefix=False,
        upstreams={},
        forward_timeout_seconds=30.0,
    )


def test_the_proxy_flags_fill_its_settings() -> None:
    app, _, _ = served_by(
        [
            "--upstream",
            "rag-reranker=http://127.0.0.1:9001",
            "--upstream",
            "rag-workflow=http://127.0.0.1:8000/",
            "--no-identity-jwt",
            "--keep-prefix",
            "--unauthenticated",
            "redirect",
            "--forward-timeout",
            "5",
        ]
    )
    assert app.state.fake_domino_proxy.settings == FakeDominoProxySettings(
        unauthenticated=UnauthenticatedResponse.REDIRECT,
        identity_jwt=False,
        keep_prefix=True,
        upstreams={
            "rag-reranker": "http://127.0.0.1:9001",
            "rag-workflow": "http://127.0.0.1:8000/",
        },
        forward_timeout_seconds=5.0,
    )


@pytest.mark.parametrize(
    "value", ["rag-reranker", "=http://x", "rag-reranker=127.0.0.1:9001"]
)
def test_an_upstream_that_is_not_name_equals_url_is_refused(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--upstream", value], serve=lambda app, host, port: None)
    assert exited.value.code == 2
    assert f"--upstream {value!r}: expected NAME=URL" in capsys.readouterr().err


def test_the_command_refuses_an_unknown_unauthenticated_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--unauthenticated", "403"], serve=lambda app, host, port: None)
    assert "invalid choice: '403'" in capsys.readouterr().err
