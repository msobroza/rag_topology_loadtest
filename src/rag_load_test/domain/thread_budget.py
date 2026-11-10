"""How a process shares its virtual CPUs between in-process model calls.

On Domino an App autoscales by starting more pods of the same hardware tier, each
a fresh process, so every process works its own budget out when it starts.

Example::

    plan_thread_budget(8, max_workers=4)   # ThreadBudget(4, 2, 2)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadBudget:
    """How many model calls run at once, and the native threads each may use."""

    workers: int  # inference calls running at once
    embedder_threads: int  # native threads one embed call may use
    reranker_threads: int  # native threads one rerank call may use


def plan_thread_budget(
    cpus: float,
    *,
    max_workers: int,
    embedder_threads: int = 0,
    reranker_threads: int = 0,
) -> ThreadBudget:
    """Split ``cpus`` so ``workers x threads`` never exceeds the whole CPUs.

    Fractional CPUs round down (half a CPU still gets one thread); a non-zero
    ``embedder_threads`` / ``reranker_threads`` replaces the computed share.

    Example::

        plan_thread_budget(8, max_workers=4, embedder_threads=1)  # (4, 1, 2)
    """
    whole = max(1, int(cpus))
    workers = min(max_workers, whole)
    share = max(1, whole // workers)
    return ThreadBudget(
        workers=workers,
        embedder_threads=embedder_threads or share,
        reranker_threads=reranker_threads or share,
    )
