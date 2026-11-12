"""The Backend names: the variants every topology allows."""

from __future__ import annotations

from rag_load_test.domain.backends import EMBEDDING_VARIANTS, RERANK_VARIANTS


def test_the_variants_load_no_model_of_ours() -> None:
    assert EMBEDDING_VARIANTS == ("api", "fake")
    assert RERANK_VARIANTS == ("api", "fake", "none")
