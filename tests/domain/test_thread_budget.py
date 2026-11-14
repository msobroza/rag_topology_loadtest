"""``plan_thread_budget``: virtual CPUs and the runtime in, workers and threads out."""

from __future__ import annotations

import pytest

from rag_load_test.domain.thread_budget import ThreadBudget, plan_thread_budget

# --- torch: the thread cap is per call, so calls share the CPUs -------------------


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [
        (8, ThreadBudget(workers=4, embedder_threads=2, reranker_threads=2)),
        (2, ThreadBudget(workers=2, embedder_threads=1, reranker_threads=1)),
        (16, ThreadBudget(workers=4, embedder_threads=4, reranker_threads=4)),
        (0.5, ThreadBudget(workers=1, embedder_threads=1, reranker_threads=1)),
        (2.5, ThreadBudget(workers=2, embedder_threads=1, reranker_threads=1)),
    ],
)
def test_torch_splits_whole_cpus_between_workers(
    cpus: float, expected: ThreadBudget
) -> None:
    assert plan_thread_budget(cpus, max_workers=4, runtime="torch") == expected


def test_a_non_zero_override_wins_for_that_model_only() -> None:
    budget = plan_thread_budget(8, max_workers=4, runtime="torch", embedder_threads=1)

    assert budget == ThreadBudget(workers=4, embedder_threads=1, reranker_threads=2)


def test_both_overrides_apply() -> None:
    budget = plan_thread_budget(
        8, max_workers=4, runtime="torch", embedder_threads=3, reranker_threads=5
    )

    assert budget == ThreadBudget(workers=4, embedder_threads=3, reranker_threads=5)


# --- openvino: one call at a time, each model on every whole CPU -------------------


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [
        (8, ThreadBudget(workers=1, embedder_threads=8, reranker_threads=8)),
        (2, ThreadBudget(workers=1, embedder_threads=2, reranker_threads=2)),
        (0.5, ThreadBudget(workers=1, embedder_threads=1, reranker_threads=1)),
        (2.5, ThreadBudget(workers=1, embedder_threads=2, reranker_threads=2)),
    ],
)
def test_openvino_runs_one_call_on_every_whole_cpu(
    cpus: float, expected: ThreadBudget
) -> None:
    assert plan_thread_budget(cpus, max_workers=4, runtime="openvino") == expected


@pytest.mark.parametrize("max_workers", [1, 4, 16])
def test_openvino_ignores_max_workers(max_workers: int) -> None:
    budget = plan_thread_budget(8, max_workers=max_workers, runtime="openvino")

    assert budget == ThreadBudget(workers=1, embedder_threads=8, reranker_threads=8)


def test_openvino_overrides_still_win() -> None:
    budget = plan_thread_budget(
        8, max_workers=4, runtime="openvino", embedder_threads=3, reranker_threads=5
    )

    assert budget == ThreadBudget(workers=1, embedder_threads=3, reranker_threads=5)
