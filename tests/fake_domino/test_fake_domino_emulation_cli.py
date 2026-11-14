"""``rag-domino-emulation``: its port, the App proxy's flags, placement files."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

from fake_domino.emulation.placement import AppPlacer
from fake_domino.emulation.proxy_settings import (
    FakeDominoProxySettings,
    UnauthenticatedResponse,
)
from fake_domino.emulation.server import main
from fake_domino.emulation.ssh_launcher import SshAppLauncher


def served_by(argv: list[str]) -> tuple[FastAPI, str, int]:
    served: list[tuple[FastAPI, str, int]] = []
    main(argv, serve=lambda app, host, port: served.append((app, host, port)))
    [(app, host, port)] = served
    return app, host, port


def test_the_emulation_serves_on_the_run_token_port_and_starts_apps() -> None:
    app, host, port = served_by([])
    assert (host, port) == ("127.0.0.1", 8899)
    assert app.state.fake_domino_apps.statuses == ["Pending", "Running"]


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


# --- placement (the emulation) ---

PLACEMENT_JSON = (
    '{"ssh": {"user": "ubuntu", "key": "/k"}, "app_port": 8888, "tiers": '
    '[{"id": "cpu-8", "cores": 8, "memory_gib": 32, "default": true, '
    '"machines": ["10.10.0.21"]}]}'
)


def test_without_placement_nothing_is_placed() -> None:
    app, _, _ = served_by([])
    assert not isinstance(app.state.fake_domino_apps.scheduler, AppPlacer)


def test_placement_and_project_env_build_an_ssh_placer(tmp_path: Path) -> None:
    placement, env = tmp_path / "placement.json", tmp_path / "project.env"
    placement.write_text(PLACEMENT_JSON)
    env.write_text("RAG_ES_PASSWORD=s3cret\n")
    app, _, _ = served_by(["--placement", str(placement), "--project-env", str(env)])
    placer = app.state.fake_domino_apps.scheduler
    assert placer.settings.default_tier.machines == ("10.10.0.21",)
    assert isinstance(placer._launcher, SshAppLauncher)
    assert placer._project_env == {"RAG_ES_PASSWORD": "s3cret"}


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"placement.json": "{}"}, "placement file {dir}/placement.json: ssh: Field"),
        (
            {"placement.json": PLACEMENT_JSON, "project.env": "A=1\nnope\n"},
            "project env file {dir}/project.env, line 2: expected KEY=VALUE",
        ),
        ({"project.env": "A=1\n"}, "--project-env needs --placement"),
    ],
)
def test_a_bad_placement_or_project_env_ends_the_command_with_its_message(
    tmp_path: Path,
    files: dict[str, str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv: list[str] = []
    for name, text in files.items():
        (tmp_path / name).write_text(text)
        flag = "--placement" if name.endswith(".json") else "--project-env"
        argv += [flag, str(tmp_path / name)]
    with pytest.raises(SystemExit) as exited:
        main(argv, serve=lambda app, host, port: None)
    assert exited.value.code == 2
    assert message.format(dir=tmp_path) in capsys.readouterr().err
