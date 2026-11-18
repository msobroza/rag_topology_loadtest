"""The ``fake`` embedder: deterministic, unit-norm, topical, and with no query prefix."""

from __future__ import annotations

import math

from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.infrastructure.fake_text import cosine


async def test_fake_embedder_is_deterministic_unit_norm_and_topical() -> None:
    embedder = FakeEmbedder()
    pie, pie_again, tart, physics = await embedder.embed_documents(
        ["red apple pie", "red apple pie", "red apple tart", "quantum gravity lecture"]
    )

    assert pie == pie_again
    assert math.isclose(math.sqrt(sum(x * x for x in pie)), 1.0, rel_tol=1e-9)
    assert cosine(pie, tart) > cosine(pie, physics)


async def test_a_query_embeds_as_the_same_text_stored_as_a_passage() -> None:
    embedder = FakeEmbedder()

    [as_a_passage] = await embedder.embed_documents(["vacation days"])

    assert await embedder.embed_query("vacation days") == as_a_passage
