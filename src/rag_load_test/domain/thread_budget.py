"""How a process shares its virtual CPUs between in-process model calls.

On Domino an App autoscales by starting more pods of the same hardware tier, each
a fresh process, so every process works its own budget out when it starts. The
split depends on the runtime: ``torch`` caps threads per call, so calls share
the CPUs; ``openvino`` fixes a model's threads when it loads and cannot overlap
two calls on one model, so one call runs at a time on every CPU.

Example:
    >>> plan_thread_budget(8, max_workers=4, runtime="torch")
    ThreadBudget(workers=4, embedder_threads=2, reranker_threads=2)
    >>> plan_thread_budget(8, max_workers=4, runtime="openvino")
    ThreadBudget(workers=1, embedder_threads=8, reranker_threads=8)
"""

from __future__ import annotations

from dataclasses import dataclass

from .backends import LocalModelRuntime


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
    runtime: LocalModelRuntime,
    embedder_threads: int = 0,
    reranker_threads: int = 0,
) -> ThreadBudget:
    """Split ``cpus`` so ``workers x threads`` never exceeds the whole CPUs.

    Fractional CPUs round down (half a CPU still gets one thread); a non-zero
    ``embedder_threads`` / ``reranker_threads`` replaces the computed share.

    Args:
        cpus: Virtual CPUs the process may use.
        max_workers: Upper bound on model calls running at once under ``torch``;
            ``openvino`` always runs one.
        runtime: What runs the models: ``torch`` or ``openvino``.
        embedder_threads: Native threads per embed call; 0 takes the computed share.
        reranker_threads: Native threads per rerank call; 0 takes the computed
            share.

    Returns:
        The workers and per-call threads.

    Example:
        >>> plan_thread_budget(8, max_workers=4, runtime="torch", embedder_threads=1)
        ThreadBudget(workers=4, embedder_threads=1, reranker_threads=2)
        >>> plan_thread_budget(8, max_workers=4, runtime="openvino")
        ThreadBudget(workers=1, embedder_threads=8, reranker_threads=8)
    """
    whole = max(1, int(cpus))
    # openvino: a loaded model holds one inference request, and a second call on
    # it while one runs raises "Infer Request is busy" (837 of them, 98% of
    # requests failed, GCP load test of 2026-11-14). One call at a time, each on
    # every CPU, removed the errors and made a rerank call faster.
    workers = 1 if runtime == "openvino" else min(max_workers, whole)
    share = max(1, whole // workers)
    return ThreadBudget(
        workers=workers,
        embedder_threads=embedder_threads or share,
        reranker_threads=reranker_threads or share,
    )
