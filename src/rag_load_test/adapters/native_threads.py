"""Scoped caps on the OpenMP threads of in-process model calls (threadpoolctl).

One scope per model: the embedder and the reranker each get their share of the
thread budget, entered in the worker thread around every model call. OpenMP
limits are per thread, so two workers can hold different limits at once. BLAS
pools are left alone: their limit is process-wide, and two scopes running at
once would overwrite each other's.

Example::

    scope = NativeThreadScope(2)              # build it after the model is loaded
    vectors = scope.run(model.encode, texts)  # at most 2 OpenMP threads inside
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class NativeThreadScope:
    """Caps the OpenMP threads of what runs inside it; one scope per model.

    ``ThreadpoolController`` snapshots the native libraries loaded when it is
    built, so build the scope after the model (and so torch) is loaded.
    """

    def __init__(self, threads: int) -> None:
        from threadpoolctl import ThreadpoolController

        self.threads = threads
        self._controller = ThreadpoolController()

    def run(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` with this thread's OpenMP pool limited to ``threads``.

        Example::

            NativeThreadScope(1).run(torch.get_num_threads)  # 1
        """
        with self._controller.limit(limits=self.threads, user_api="openmp"):
            return fn(*args)


def settle_torch_thread() -> None:
    """Executor initializer: lets torch finish its per-thread setup.

    torch sets its own per-thread thread count on a thread's FIRST torch call,
    which undoes a scope entered before it (measured with torch 2.15: the first
    scoped call on a fresh worker ran with 4 threads instead of 1). Making that
    first call here, before any scope, closes the gap. torch is only touched when
    it is already loaded, so processes with fake models never import it.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.get_num_threads()
