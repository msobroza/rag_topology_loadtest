"""The suite cannot reach real model weights: ``conftest.py`` shuts the Hub and its cache.

A test that opens an in-process model without a fake must fail at once. It
downloads the model otherwise, or loads a cached one and exports it, which
filled the disk twice on 2026-11-15.
"""

from __future__ import annotations

import os
from pathlib import Path


def test_the_hub_is_offline_for_every_test() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_the_model_cache_the_tests_see_holds_no_model() -> None:
    cache = Path(os.environ["HF_HUB_CACHE"])

    assert cache.parent == Path(os.environ["HF_HOME"])
    assert not cache.exists() or not any(cache.iterdir())
