"""The one bounded thread pool the in-process models run on.

sentence-transformers is synchronous, so model calls go through ``run_blocking``
on one bounded ``ThreadPoolExecutor``: the event loop stays free and overload
turns into queueing.

Example:
    >>> pool = model_executor(4, on_thread_start=settle_torch_thread)
    >>> vectors = await run_blocking(pool, model.encode, texts)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def model_executor(
    workers: int, on_thread_start: Callable[[], None] | None = None
) -> ThreadPoolExecutor:
    """Build the shared pool for sentence-transformers inference.

    ``workers`` calls run at once (from the thread budget), the rest queue.

    Args:
        workers: Inference calls that run at once.
        on_thread_start: Runs once in each new worker thread
            (``settle_torch_thread`` for real models); ``None`` runs nothing.

    Returns:
        The executor, its threads named ``rag-model``.

    Example:
        >>> pool = model_executor(4, on_thread_start=settle_torch_thread)
    """
    return ThreadPoolExecutor(
        workers, thread_name_prefix="rag-model", initializer=on_thread_start
    )


async def run_blocking(executor: Executor, fn: Callable[..., T], *args: Any) -> T:
    """Await ``fn(*args)`` on ``executor`` (``functools.partial`` for kwargs).

    Args:
        executor: The pool the blocking call runs on.
        fn: The blocking call.
        *args: Positional arguments passed to ``fn``.

    Returns:
        What ``fn`` returns.
    """
    return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)
