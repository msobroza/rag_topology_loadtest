"""How many virtual CPUs this process may use, and the thread budget from them.

Domino gives a run no CPU-count variable, only ``DOMINO_HARDWARE_TIER_ID`` and the
credential-free ``DOMINO_API_PROXY``; the tier's ``resources.cores`` (the
Kubernetes CPU request, possibly fractional) comes from the Hardware Tiers API.
Outside Domino, ``joblib.cpu_count()`` reads the container's CPU quota and
affinity, which ``os.cpu_count()`` ignores. Counters are asked in order and the
first that answers wins; ``RagPorts`` decides the order.

Example:
    >>> counters = [FixedCpus(0), DominoHardwareTierCpus(http, proxy, tier, policy),
    ...             ContainerCpus()]
    >>> budget = await work_out_thread_budget(counters, max_workers=4)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from http import HTTPStatus
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
    async def cpus(self) -> float | None:
        """Ask this source for the CPU count.

        Returns:
            The virtual CPUs, possibly fractional; ``None`` when this source cannot
            say, so the next counter is asked.
        """


class FixedCpus(CpuCounter):
    """``RAG_MODEL_CPUS``: answers only when positive (0 = find out).

    Args:
        cpus: The configured CPU count; 0 or less leaves it to the next counter.
    """

    name = "RAG_MODEL_CPUS"

    def __init__(self, cpus: float) -> None:
        self._cpus = cpus

    async def cpus(self) -> float | None:
        """Return the configured count.

        Returns:
            The configured CPUs when positive, else ``None``.
        """
        return self._cpus if self._cpus > 0 else None


class ContainerCpus(CpuCounter):
    """The CPUs the container may use (cgroup quota and affinity honoured)."""

    name = "container"

    async def cpus(self) -> float | None:
        """Ask ``joblib.cpu_count()``, which always answers.

        Returns:
            The CPUs the container's quota and affinity allow.
        """
        import joblib  # only processes with in-process models ask

        return float(joblib.cpu_count())


class DominoHardwareTierCpus(CpuCounter):
    """``resources.cores`` of the run's hardware tier, via the Domino API proxy.

    Any failure (no ``ViewHardwareTiers`` permission -> 403, a bad body, no
    proxy) logs one JSON warning and answers ``None``: it must never stop the app.

    Args:
        http: The shared HTTP client the tier is fetched with.
        api_proxy: Base URL of the credential-free Domino API proxy
            (``DOMINO_API_PROXY``).
        tier_id: The run's hardware tier (``DOMINO_HARDWARE_TIER_ID``).
        policy: Timeout, retries and backoff for the tier request.

    Example:
        >>> await DominoHardwareTierCpus(http, os.environ["DOMINO_API_PROXY"],
        ...                              "small-k8s", policy).cpus()
        2.0
    """

    name = "domino_hardware_tier"

    def __init__(
        self, http: httpx.AsyncClient, api_proxy: str, tier_id: str, policy: HttpPolicy
    ) -> None:
        self._http, self._tier_id, self._policy = http, tier_id, policy
        self._url = api_proxy.rstrip("/") + TIER_PATH.format(tier_id=tier_id)

    async def cpus(self) -> float | None:
        """Fetch the tier's ``resources.cores``, logging a warning on any failure.

        Returns:
            The tier's cores when positive; ``None`` on a transport error, a non-200
            status or a body without positive ``resources.cores``.
        """
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
        if response.status_code != HTTPStatus.OK:
            return None, {"status": response.status_code}
        cores = _cores(response)
        if cores is None:
            return None, {"status": 200, "error": "no positive resources.cores"}
        return cores, {}

    # GET the tier, retrying transport errors as the policy says.
    async def _get(self) -> httpx.Response:
        last = self._policy.retries
        for attempt in range(last):
            with contextlib.suppress(httpx.TransportError):
                return await self._get_after_backoff(attempt)
        return await self._get_after_backoff(last)  # its transport error propagates

    async def _get_after_backoff(self, attempt: int) -> httpx.Response:
        await asyncio.sleep(self._policy.backoff_before(attempt))
        return await self._http.get(self._url, timeout=self._policy.timeout_s)


# ``hardwareTier.resources.cores`` when it is a positive number, else None.
def _cores(response: httpx.Response) -> float | None:
    try:
        cores = response.json()["hardwareTier"]["resources"]["cores"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(cores, bool) or not isinstance(cores, (int, float)):
        return None
    return float(cores) if cores > 0 else None


async def first_cpu_count(counters: Sequence[CpuCounter]) -> tuple[float, str]:
    """Ask the counters in order and return the first answer.

    Args:
        counters: The CPU sources, asked in order (``RagPorts`` decides it).

    Returns:
        ``(cpus, source name)`` from the first counter that answers.

    Raises:
        RuntimeError: If no counter answers; the message names every counter asked.

    Example:
        >>> await first_cpu_count([FixedCpus(0), ContainerCpus()])
        (8.0, 'container')
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

    Args:
        counters: The CPU sources, asked in order; the first answer wins.
        max_workers: Upper bound on model calls running at once.
        embedder_threads: Native threads per embed call; 0 takes the computed share.
        reranker_threads: Native threads per rerank call; 0 takes the computed
            share.

    Returns:
        The workers and per-call threads.

    Example:
        >>> await work_out_thread_budget([FixedCpus(8)], max_workers=4)
        ThreadBudget(workers=4, embedder_threads=2, reranker_threads=2)
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
