"""The text helpers of the ``fake`` backends: cosine on vectors of any length."""

from __future__ import annotations

import pytest

from rag_load_test.infrastructure.fake_text import cosine


def test_cosine_ignores_the_lengths_of_the_vectors() -> None:
    assert cosine([3.0, 0.0], [0.5, 0.0]) == pytest.approx(1.0)
    assert cosine([2.0, 2.0], [0.0, 7.0]) == pytest.approx(2**-0.5)
    assert cosine([1.0, 0.0], [-4.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_with_a_zero_vector_is_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_cosine_rejects_vectors_of_different_dimensions() -> None:
    with pytest.raises(ValueError, match="shorter"):  # zip(strict=True)
        cosine([1.0, 0.0], [1.0])
