"""Where the CPU count comes from: the Domino hardware tier, the container, or a fixed value."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from cpu_fakes import FakeCpuCounter
from rag_load_test.adapters.cpu_count import (
    ContainerCpus,
    DominoHardwareTierCpus,
    FixedCpus,
    first_cpu_count,
)
from rag_load_test.adapters.http_policy import HttpPolicy

PROXY = "http://proxy.test:8899"
TIER_PATH = "/api/hardwaretiers/v1/hardwaretiers/tier-9"
NO_WAIT = HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.0)
Handler = Callable[[httpx.Request], httpx.Response]


def tier_body(resources: dict[str, object]) -> httpx.Response:
    envelope = {"hardwareTier": {"id": "tier-9", "resources": resources}}
    return httpx.Response(200, json={**envelope, "metadata": {}})


@pytest.fixture
async def tier_cpus() -> AsyncIterator[Callable[[Handler], DominoHardwareTierCpus]]:
    clients: list[httpx.AsyncClient] = []

    def make(handler: Handler) -> DominoHardwareTierCpus:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return DominoHardwareTierCpus(client, PROXY + "/", "tier-9", NO_WAIT)

    yield make
    for client in clients:
        await client.aclose()


def warnings_logged(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [json.loads(r.message) for r in caplog.records if r.levelname == "WARNING"]


# --- DominoHardwareTierCpus -----------------------------------------------------


async def test_tier_cores_are_the_cpu_count(
    tier_cpus: Callable[[Handler], DominoHardwareTierCpus],
) -> None:
    paths: list[str] = []

    def domino(request: httpx.Request) -> httpx.Response:
        paths.append(str(request.url))
        return tier_body({"cores": 2.5, "coresLimit": 4.0, "memory": {"value": 8}})

    assert await tier_cpus(domino).cpus() == 2.5
    assert paths == [PROXY + TIER_PATH]


async def test_a_403_gives_none_and_one_warning(
    tier_cpus: Callable[[Handler], DominoHardwareTierCpus],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="missing ViewHardwareTiers")

    with caplog.at_level("WARNING", logger="rag_load_test"):
        assert await tier_cpus(forbidden).cpus() is None

    [warning] = warnings_logged(caplog)
    assert warning["event"] == "hardware_tier_cpus_unavailable"
    assert (warning["tier_id"], warning["status"]) == ("tier-9", 403)


async def test_a_connect_error_is_retried_then_gives_none(
    tier_cpus: Callable[[Handler], DominoHardwareTierCpus],
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts: list[int] = []

    def refused(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("refused")

    with caplog.at_level("WARNING", logger="rag_load_test"):
        assert await tier_cpus(refused).cpus() is None

    assert len(attempts) == 3  # the first attempt and NO_WAIT.retries=2
    [warning] = warnings_logged(caplog)
    assert warning["error"] == "ConnectError: refused"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>login</html>"),
        httpx.Response(200, json={"hardwareTier": {"resources": {}}}),
        tier_body({"cores": 0}),
        tier_body({"cores": "two"}),
    ],
    ids=["not-json", "no-cores", "zero-cores", "text-cores"],
)
async def test_an_unusable_body_gives_none_and_one_warning(
    tier_cpus: Callable[[Handler], DominoHardwareTierCpus],
    caplog: pytest.LogCaptureFixture,
    response: httpx.Response,
) -> None:
    with caplog.at_level("WARNING", logger="rag_load_test"):
        assert await tier_cpus(lambda request: response).cpus() is None

    assert len(warnings_logged(caplog)) == 1


# --- the other counters and the order ---------------------------------------------


async def test_fixed_cpus_answer_only_when_positive() -> None:
    assert await FixedCpus(3.0).cpus() == 3.0
    assert await FixedCpus(0.0).cpus() is None


async def test_container_cpus_are_a_positive_whole_number() -> None:
    cpus = await ContainerCpus().cpus()
    assert cpus is not None and cpus >= 1 and cpus == int(cpus)


async def test_the_first_counter_that_answers_wins() -> None:
    silent, tier, never = (
        FakeCpuCounter(None, "a"),
        FakeCpuCounter(6, "b"),
        FakeCpuCounter(2, "c"),
    )

    assert await first_cpu_count([silent, tier, never]) == (6, "b")
    assert (silent.asked, tier.asked, never.asked) == (1, 1, 0)


async def test_no_counter_answering_is_an_error_naming_them() -> None:
    with pytest.raises(RuntimeError, match="a, b"):
        await first_cpu_count([FakeCpuCounter(None, "a"), FakeCpuCounter(None, "b")])
