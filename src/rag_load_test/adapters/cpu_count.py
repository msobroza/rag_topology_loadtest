"""How many virtual CPUs this process may use, and the thread budget made from them.

Domino gives a run no CPU-count variable, only ``DOMINO_HARDWARE_TIER_ID`` and the
credential-free ``DOMINO_API_PROXY``; the tier's ``resources.cores`` (the
Kubernetes CPU request, possibly fractional) comes from the Hardware Tiers API.
Outside Domino, ``joblib.cpu_count()`` reads the container's CPU quota and
affinity, which ``os.cpu_count()`` ignores. Counters are asked in order and the
first that answers wins; ``RagPorts`` decides the order.

Example::

    counters = [FixedCpus(0), DominoHardwareTierCpus(http, proxy, tier, policy),
                ContainerCpus()]
    budget = await work_out_thread_budget(counters, max_workers=4)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from ..domain.errors import describe_error
from ..domain.thread_budget import ThreadBudget, plan_thread_budget
from .http_policy import HttpPolicy
from .json_log import log_json

TIER_PATH = "/api/hardwaretiers/v1/hardwaretiers/{tier_id}"


class CpuCounter(ABC):
    """One source of the CPU count; ``None`` means this source cannot say."""

    name: str  # for the log line

    @abstractmethod
    async def cpus(self) -> float | None: ...


class FixedCpus(CpuCounter):
    """``RAG_MODEL_CPUS``: answers only when positive (0 = find out)."""

    name = "RAG_MODEL_CPUS"

    def __init__(self, cpus: float) -> None:
        self._cpus = cpus

    async def cpus(self) -> float | None:
        return self._cpus if self._cpus > 0 else None


class ContainerCpus(CpuCounter):
    """The CPUs the container may use (cgroup quota and affinity honoured)."""

    name = "container"

    async def cpus(self) -> float | None:
        import joblib  # only processes with in-process models ask

        return float(joblib.cpu_count())


class DominoHardwareTierCpus(CpuCounter):
    """``resources.cores`` of the run's hardware tier, via the Domino API proxy.

    Any failure (no ``ViewHardwareTiers`` permission -> 403, a bad body, no
    proxy) logs one JSON warning and answers ``None``: it must never stop the app.

    Example::

        await DominoHardwareTierCpus(http, os.environ["DOMINO_API_PROXY"],
                                     "small-k8s", policy).cpus()   # 2.0
    """

    name = "domino_hardware_tier"

    def __init__(
        self, http: httpx.AsyncClient, api_proxy: str, tier_id: str, policy: HttpPolicy
    ) -> None:
        self._http, self._tier_id, self._policy = http, tier_id, policy
        self._url = api_proxy.rstrip("/") + TIER_PATH.format(tier_id=tier_id)

    async def cpus(self) -> float | None:
        cores, problem = await self._cores_or_problem()
        if problem:
            log_json(
                logging.WARNING,
                event="hardware_tier_cpus_unavailable",
                tier_id=self._tier_id,
                url=self._url,
                **problem,
            )
        return cores

    async def _cores_or_problem(self) -> tuple[float | None, dict[str, Any]]:
        try:
            response = await self._get()
        except httpx.TransportError as exc:
            return None, {"error": describe_error(exc)}
        if response.status_code != 200:
            return None, {"status": response.status_code}
        cores = _cores(response)
        if cores is None:
            return None, {"status": 200, "error": "no positive resources.cores"}
        return cores, {}

    async def _get(self) -> httpx.Response:
        """GET the tier, retrying transport errors as the policy says."""
        for attempt in range(self._policy.retries + 1):
            await asyncio.sleep(self._policy.backoff_before(attempt))
            try:
                return await self._http.get(self._url, timeout=self._policy.timeout_s)
            except httpx.TransportError:
                if attempt == self._policy.retries:
                    raise
        raise AssertionError("unreachable: the loop returns or raises")


def _cores(response: httpx.Response) -> float | None:
    """``hardwareTier.resources.cores`` when it is a positive number, else None."""
    try:
        cores = response.json()["hardwareTier"]["resources"]["cores"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(cores, bool) or not isinstance(cores, (int, float)):
        return None
    return float(cores) if cores > 0 else None


async def first_cpu_count(counters: Sequence[CpuCounter]) -> tuple[float, str]:
    """``(cpus, source name)`` from the first counter that answers.

    Example::

        await first_cpu_count([FixedCpus(0), ContainerCpus()])  # (8.0, "container")
    """
    for counter in counters:
        cpus = await counter.cpus()
        if cpus is not None:
            return cpus, counter.name
    names = ", ".join(counter.name for counter in counters)
    raise RuntimeError(f"no CPU counter answered; asked: {names}")


async def work_out_thread_budget(
    counters: Sequence[CpuCounter],
    *,
    max_workers: int,
    embedder_threads: int = 0,
    reranker_threads: int = 0,
) -> ThreadBudget:
    """Count the CPUs, plan the budget, and log one ``thread_budget`` JSON line.

    Example::

        await work_out_thread_budget([FixedCpus(8)], max_workers=4)  # (4, 2, 2)
    """
    cpus, source = await first_cpu_count(counters)
    budget = plan_thread_budget(
        cpus,
        max_workers=max_workers,
        embedder_threads=embedder_threads,
        reranker_threads=reranker_threads,
    )
    log_json(
        logging.INFO,
        event="thread_budget",
        cpus=cpus,
        source=source,
        workers=budget.workers,
        embedder_threads=budget.embedder_threads,
        reranker_threads=budget.reranker_threads,
    )
    return budget
