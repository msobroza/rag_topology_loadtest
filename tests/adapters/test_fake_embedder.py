"""The ``fake`` embedder: deterministic, unit-norm, topical."""

from __future__ import annotations

import math

from rag_load_test.adapters.embedder.fake.fake_embedder import FakeEmbedder
from rag_load_test.infrastructure.fake_text import cosine


async def test_fake_embedder_is_deterministic_unit_norm_and_topical() -> None:
    embedder = FakeEmbedder()
    pie, pie_again, tart, physics = await embedder.embed(
        ["red apple pie", "red apple pie", "red apple tart", "quantum gravity lecture"]
    )

    assert pie == pie_again
    assert math.isclose(math.sqrt(sum(x * x for x in pie)), 1.0, rel_tol=1e-9)
    assert cosine(pie, tart) > cosine(pie, physics)
