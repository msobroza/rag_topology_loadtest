"""``NativeThreadScope`` against real torch (no weights): limits hold per worker thread.

On some builds torch sets its own per-thread OpenMP default on a thread's FIRST
torch call, which undoes a scope entered before it; ``settle_torch_thread`` as the
executor initializer makes that first call happen before any scope. These tests
run each check on a worker's first call, the case that reset breaks. The reset
itself is not asserted: it is torch's behaviour, seen on macOS arm64 and absent on
Linux x86, so a test of it would fail on one platform or the other.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

from rag_load_test.adapters.native_threads import (  # noqa: E402
    NativeThreadScope,
    settle_torch_thread,
)


def test_run_returns_the_result_and_passes_the_arguments() -> None:
    scope = NativeThreadScope(2)
    assert scope.threads == 2
    assert scope.run(divmod, 7, 2) == (3, 1)


def test_the_first_call_on_a_settled_worker_honours_the_scope() -> None:
    scope = NativeThreadScope(1)
    with ThreadPoolExecutor(1, initializer=settle_torch_thread) as pool:
        first = pool.submit(scope.run, torch.get_num_threads).result()
        second = pool.submit(scope.run, torch.get_num_threads).result()

    assert (first, second) == (1, 1)


def test_two_workers_hold_different_limits_at_the_same_time() -> None:
    both_inside = threading.Barrier(2, timeout=10)

    def threads_while_both_are_scoped() -> int:
        both_inside.wait()  # the other worker is inside its own scope now
        threads = torch.get_num_threads()
        both_inside.wait()
        return threads

    with ThreadPoolExecutor(2, initializer=settle_torch_thread) as pool:
        one = pool.submit(NativeThreadScope(1).run, threads_while_both_are_scoped)
        three = pool.submit(NativeThreadScope(3).run, threads_while_both_are_scoped)
        assert (one.result(), three.result()) == (1, 3)


def test_a_scope_leaves_the_process_wide_blas_pools_alone() -> None:
    """A BLAS limit is process-wide: two scopes at once would overwrite each other's."""
    pytest.importorskip("numpy")  # brings its BLAS along where it ships one
    from threadpoolctl import threadpool_info

    def blas_threads() -> list[int]:
        pools = threadpool_info()
        return [pool["num_threads"] for pool in pools if pool["user_api"] == "blas"]

    before = blas_threads()
    if not before or max(before) < 2:
        pytest.skip("no multi-threaded BLAS pool is loaded here (numpy on Accelerate)")

    assert NativeThreadScope(1).run(blas_threads) == before
