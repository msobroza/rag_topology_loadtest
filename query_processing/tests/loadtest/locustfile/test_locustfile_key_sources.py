"""``locustfile.py`` and the API key each request carries: its key source, a 401, gevent.

The users, the stand-in response and the import of the Locust file are
``test_locustfile``'s.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from test_locustfile import (
    LOCUSTFILE,
    FakeCaughtResponse,
    FakeLocustUser,
    _import_locustfile,
)

from loadtest_settings_helpers import clear_loadtest_variables, settings_from
from query_processing.loadtest.request_plans import RequestPlan
from query_processing.loadtest.target_client import TargetKey, opened_target_key


@pytest.fixture(autouse=True)
def no_loadtest_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_loadtest_variables(monkeypatch)


@pytest.fixture
def locustfile() -> ModuleType:
    return _import_locustfile("locustfile_key_sources_under_test")


TOKEN_URL = "http://domino.test/access-token"
RETRIEVE = RequestPlan("/retrieve", {"question": "q"}, "/retrieve")


class FakeTokenService:
    """Domino's token URL as a stand-in transport: ``run-token-1``, ``-2``, ..."""

    def __init__(self) -> None:
        self.served = 0
        self.transport = httpx.MockTransport(self._serve)

    def _serve(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == TOKEN_URL
        self.served += 1
        return httpx.Response(200, text=f"run-token-{self.served}")


class FakeLocustTarget(FakeLocustUser):
    """A user whose target refuses the ``Authorization`` values in ``refused``."""

    def __init__(self, refused: set[str] | None = None) -> None:
        super().__init__(FakeCaughtResponse(200, "{}"))
        self.responses: list[FakeCaughtResponse] = []
        self._refused = refused or set()

    def _post(self, path: str, **kwargs: Any) -> FakeCaughtResponse:
        self.posted.append({"path": path, **kwargs})
        refused = kwargs["headers"].get("Authorization") in self._refused
        response = FakeCaughtResponse(401 if refused else 200, "{}")
        self.responses.append(response)
        return response

    def carried(self) -> list[str | None]:
        return [posted["headers"].get("Authorization") for posted in self.posted]


@contextmanager
def domino_key(
    monkeypatch: pytest.MonkeyPatch, tokens: FakeTokenService
) -> Generator[TargetKey, None, None]:
    """The load test's key with ``QP_LOADTEST_KEY__SOURCE=domino``, on ``tokens``."""
    variables = {"QP_LOADTEST_KEY__SOURCE": "domino", "QP_DOMINO__TOKEN_URL": TOKEN_URL}
    with opened_target_key(
        settings_from(monkeypatch, variables), tokens.transport
    ) as key:
        yield key


@pytest.mark.parametrize(
    ("variables", "header", "value"),
    [
        (
            {"QP_LOADTEST_KEY__SOURCE": "env", "QP_LOADTEST_KEY__VARIABLE": "LT_KEY"},
            "Authorization",
            "Bearer k-1",
        ),
        ({"DOMINO_API_KEY": "dk"}, "X-Domino-Api-Key", "dk"),
        (
            {
                "QP_LOADTEST_KEY__SOURCE": "env",
                "QP_LOADTEST_KEY__VARIABLE": "LT_KEY",
                "DOMINO_API_KEY": "dk",
            },
            "Authorization",
            "Bearer k-1",
        ),
    ],
    ids=["env", "domino-api-key", "a-key-block-wins"],
)
def test_each_request_carries_the_key_the_settings_name(
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    header: str,
    value: str,
) -> None:
    monkeypatch.setenv("LT_KEY", "k-1")
    for name, given in variables.items():
        monkeypatch.setenv(name, given)
    module = _import_locustfile(f"locustfile_key_{header}_{len(variables)}")
    user = FakeLocustTarget()

    module.RagUser.retrieve(user)
    module.RagUser.query(user)

    assert [p["headers"] for p in user.posted] == [{header: value}] * 2


def test_each_request_carries_the_run_token_held_for_its_lifetime(
    locustfile: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens, user = FakeTokenService(), FakeLocustTarget()
    with domino_key(monkeypatch, tokens) as key:
        for _ in range(3):
            locustfile.send(user, RETRIEVE, key)

    assert tokens.served == 1  # within QP_DOMINO__TOKEN_TTL_S's 240 s
    assert user.carried() == ["Bearer run-token-1"] * 3


def test_a_401_is_one_failure_not_sent_again_and_the_next_request_has_a_new_token(
    locustfile: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = FakeTokenService()
    user = FakeLocustTarget(refused={"Bearer run-token-1"})
    with domino_key(monkeypatch, tokens) as key:
        locustfile.send(user, RETRIEVE, key)
        locustfile.send(user, RETRIEVE, key)

    first, second = user.responses
    assert first.failures == ["HTTP 401: {}"]  # recorded, and sent once
    assert (second.successes, second.failures) == (1, [])
    assert user.carried() == ["Bearer run-token-1", "Bearer run-token-2"]
    assert tokens.served == 2


# Five Locust users, greenlets in a process gevent patched as Locust patches it,
# send at once with no token held. The token URL is a stand-in transport that
# yields to the others while it answers, as an ask on the network would.
UNDER_GEVENT = r"""
import importlib.util, json, sys
from types import SimpleNamespace
import gevent, httpx
from query_processing.loadtest.request_plans import RequestPlan
from query_processing.loadtest.target_client import opened_target_key

spec = importlib.util.spec_from_file_location("locustfile_under_gevent", sys.argv[1])
locustfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locustfile)  # imports locust, which patches the interpreter
served, carried = [], []

def serve(request):
    gevent.sleep(0.01)
    served.append(str(request.url))
    return httpx.Response(200, text=f"run-token-{len(served)}")

class Answered:  # what Locust's catch_response hands send: a 200
    status_code, text, headers = 200, "{}", {}
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def success(self): pass
    def failure(self, message): raise AssertionError(message)

def post(path, **kwargs):
    carried.append(kwargs["headers"]["Authorization"])
    return Answered()

user = SimpleNamespace(client=SimpleNamespace(post=post))
plan = RequestPlan("/retrieve", {"question": "q"}, "/retrieve")
with opened_target_key(locustfile.SETTINGS, httpx.MockTransport(serve)) as key:
    users = [gevent.spawn(locustfile.send, user, plan, key) for _ in range(5)]
    gevent.joinall(users, raise_error=True)
print(json.dumps({"patched": gevent.monkey.is_module_patched("threading"),
                  "served": served, "carried": carried}))
"""


def test_users_under_gevent_that_find_no_token_share_one_ask(tmp_path: Path) -> None:
    # The suite keeps gevent out (tests/conftest.py), so the Locust file runs
    # in a process of its own, patched: the source's locks must let the other
    # users run while one asks, and those users must take its token.
    env = {k: v for k, v in os.environ.items() if k != "LOCUST_SKIP_MONKEY_PATCH"}
    env.update({"QP_LOADTEST_KEY__SOURCE": "domino", "QP_DOMINO__TOKEN_URL": TOKEN_URL})
    command = [sys.executable, "-c", UNDER_GEVENT, str(LOCUSTFILE)]

    # The timeout only stops a deadlock from hanging the suite.
    run = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )

    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout.splitlines()[-1])
    assert seen["patched"] is True
    assert seen["served"] == [TOKEN_URL]
    assert seen["carried"] == ["Bearer run-token-1"] * 5


def test_the_key_source_opened_at_import_is_closed_as_locust_quits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_LOADTEST_KEY__SOURCE", "domino")
    monkeypatch.setenv("QP_DOMINO__TOKEN_URL", TOKEN_URL)
    module = _import_locustfile("locustfile_closed_key")  # asks nothing yet

    module.close_key_source_on_quit()

    with pytest.raises(RuntimeError, match="client has been closed"):
        module.KEY.headers()


def test_a_key_block_start_up_would_refuse_fails_the_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_LOADTEST_KEY__SOURCE", "vault")

    with pytest.raises(ValueError, match="QP_LOADTEST_KEY__SOURCE: unknown key"):
        _import_locustfile("locustfile_unknown_source")
