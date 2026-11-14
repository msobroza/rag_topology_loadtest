"""``ResourceScope``: each resource opened once, all closed together, failures named."""

from __future__ import annotations

import pytest

from rag_load_test.composition.resource_scope import OWNED, ResourceScope
from rag_load_test.domain.errors import RagDependencyError


class CountingOpener:
    """An opener that returns a new object each time and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> object:
        self.calls += 1
        return object()


async def test_a_resource_opens_once_and_is_reused_until_the_block_ends() -> None:
    open_it = CountingOpener()
    async with ResourceScope("Owner") as scope:
        first = await scope.once("thing", OWNED, open_it)
        second = await scope.once("thing", OWNED, open_it)

    assert first is second
    assert open_it.calls == 1


async def test_two_names_are_two_resources() -> None:
    open_it = CountingOpener()
    async with ResourceScope("Owner") as scope:
        one = await scope.once("one", OWNED, open_it)
        other = await scope.once("other", OWNED, open_it)

    assert one is not other


async def test_what_was_registered_closes_in_reverse_when_the_block_ends() -> None:
    closed: list[str] = []

    async def close_second() -> None:
        closed.append("second")

    async with ResourceScope("Owner") as scope:
        scope.exit_stack("first").callback(closed.append, "first")
        scope.exit_stack("second").push_async_callback(close_second)
        assert closed == []

    assert closed == ["second", "first"]


async def test_a_new_block_opens_everything_again() -> None:
    scope, open_it = ResourceScope("Owner"), CountingOpener()
    async with scope:
        before = await scope.once("thing", OWNED, open_it)
    async with scope:
        after = await scope.once("thing", OWNED, open_it)

    assert before is not after
    assert open_it.calls == 2


async def test_a_failure_to_open_names_the_resource_and_its_backend() -> None:
    async def broken() -> object:
        raise OSError("disk full")

    async with ResourceScope("Owner") as scope:
        with pytest.raises(RagDependencyError) as caught:
            await scope.once("llm", "api", broken)

    assert str(caught.value) == (
        "llm unavailable: could not open backend 'api': OSError: disk full"
    )
    assert isinstance(caught.value.__cause__, OSError)


async def test_a_dependency_error_from_further_down_is_not_wrapped() -> None:
    inner = RagDependencyError("local_reranker", "no weights")

    async def broken() -> object:
        raise inner

    async with ResourceScope("Owner") as scope:
        with pytest.raises(RagDependencyError) as caught:
            await scope.once("reranker", "local", broken)

    assert caught.value is inner


async def test_a_resource_that_failed_is_tried_again_on_the_next_request() -> None:
    attempts: list[int] = []

    async def fails_once() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("not yet")
        return "opened"

    async with ResourceScope("Owner") as scope:
        with pytest.raises(RagDependencyError):
            await scope.once("thing", OWNED, fails_once)
        assert await scope.once("thing", OWNED, fails_once) == "opened"


async def test_a_request_outside_the_block_names_the_owner_and_the_resource() -> None:
    scope = ResourceScope("CompositionRoot")
    expected = (
        "CompositionRoot: 'http_client' requested outside "
        r"'async with CompositionRoot\(\.\.\.\)'"
    )

    with pytest.raises(RuntimeError, match=expected):
        await scope.once("http_client", OWNED, CountingOpener())
    with pytest.raises(RuntimeError, match=expected):
        scope.exit_stack("http_client")


async def test_entering_an_open_scope_is_refused() -> None:
    async with ResourceScope("CompositionRoot") as scope:
        with pytest.raises(RuntimeError, match="CompositionRoot is already open"):
            await scope.__aenter__()
