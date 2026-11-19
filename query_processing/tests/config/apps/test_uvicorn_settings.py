"""What a process of ``qp-serve`` or ``qp-model-server`` runs as: Uvicorn's four settings.

Each default is Uvicorn's own, so an App that sets nothing runs as it always
has; the workers count is always passed, so ``WEB_CONCURRENCY`` changes nothing.
"""

from __future__ import annotations

import pytest
import uvicorn
from pydantic import ValidationError

from query_processing.config.apps.uvicorn_settings import UvicornSettings


def test_the_defaults_are_uvicorns_own() -> None:
    config = uvicorn.Config("x:y")

    assert UvicornSettings(_env_file=None).run_arguments() == {
        "workers": 1,
        "limit_concurrency": config.limit_concurrency,
        "backlog": config.backlog,
        "timeout_keep_alive": config.timeout_keep_alive,
    }


def test_the_variables_set_uvicorns_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP_UVICORN_WORKERS", "2")
    monkeypatch.setenv("QP_UVICORN_LIMIT_CONCURRENCY", "64")
    monkeypatch.setenv("QP_UVICORN_BACKLOG", "512")
    monkeypatch.setenv("QP_UVICORN_TIMEOUT_KEEP_ALIVE_S", "30")

    assert UvicornSettings(_env_file=None).run_arguments() == {
        "workers": 2,
        "limit_concurrency": 64,
        "backlog": 512,
        "timeout_keep_alive": 30,
    }


def test_web_concurrency_in_the_environment_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "4")

    assert UvicornSettings(_env_file=None).run_arguments()["workers"] == 1


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("QP_UVICORN_WORKERS", "0"),
        ("QP_UVICORN_LIMIT_CONCURRENCY", "-1"),
        ("QP_UVICORN_BACKLOG", "0"),
        ("QP_UVICORN_TIMEOUT_KEEP_ALIVE_S", "-1"),
    ],
)
def test_a_value_out_of_range_is_refused_naming_its_field(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError) as refused:
        UvicornSettings(_env_file=None)

    field = variable.removeprefix("QP_UVICORN_").lower()
    assert field in str(refused.value)
    assert UvicornSettings.variable_of(field) == variable


def test_variable_of_refuses_a_name_that_is_no_field() -> None:
    with pytest.raises(ValueError, match="'threads'"):
        UvicornSettings.variable_of("threads")
