"""Fixtures of the ``local`` embedder and reranker tests: the pool, a fake library.

Loaded with ``pytest_plugins = ["local_model_fixtures"]`` in a test module, under
this name so pytest can rewrite its asserts; the doubles are in
``local_model_fakes.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_model_fakes import FakeLoadingModule
from rag_load_test.adapters.local_models.model_pool import model_executor


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    """The model pool, two workers, shut down after the test."""
    pool = model_executor(workers=2)
    yield pool
    pool.shutdown(wait=True)


@pytest.fixture
def fake_loading(monkeypatch: pytest.MonkeyPatch) -> FakeLoadingModule:
    module = FakeLoadingModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module
