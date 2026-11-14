"""The one bounded thread pool the in-process models run on."""

from __future__ import annotations

import threading

from rag_load_test.infrastructure.local_models.model_pool import model_executor


def test_model_executor_runs_the_given_number_of_workers() -> None:
    pool = model_executor(workers=3)
    try:
        assert pool._max_workers == 3
    finally:
        pool.shutdown(wait=True)


def test_model_executor_runs_on_thread_start_in_each_new_worker() -> None:
    started: list[str] = []

    def record() -> None:
        started.append(threading.current_thread().name)

    pool = model_executor(workers=1, on_thread_start=record)
    try:
        worker = pool.submit(lambda: threading.current_thread().name).result()
    finally:
        pool.shutdown(wait=True)
    assert started == [worker]
    assert worker.startswith("rag-model")
