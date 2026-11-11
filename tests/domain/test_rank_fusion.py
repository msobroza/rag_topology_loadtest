"""``reciprocal_rank_fusion``: one ranking out of several, by rank alone."""

from __future__ import annotations

import pytest

from rag_load_test.domain.passages import ScoredPassage
from rag_load_test.domain.rank_fusion import reciprocal_rank_fusion


def hit(passage_id: str, score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=passage_id, text=passage_id, retrieval_score=score)


def ids(passages: list[ScoredPassage]) -> list[str]:
    return [passage.id for passage in passages]


def test_a_passage_in_both_rankings_beats_one_ranked_first_in_one() -> None:
    dense = [hit("a"), hit("b"), hit("c")]
    lexical = [hit("b"), hit("d")]

    fused = reciprocal_rank_fusion([dense, lexical], k=60, top_k=4)

    assert ids(fused) == ["b", "a", "d", "c"]


def test_the_fused_score_is_the_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("b")]], k=60, top_k=2)

    scores = {passage.id: passage.retrieval_score for passage in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)


def test_only_ranks_count_not_the_rankings_own_scores() -> None:
    # A BM25 score of 40 and a cosine of 0.9 are not comparable; their ranks are.
    dense = [hit("a", 0.9), hit("b", 0.1)]
    lexical = [hit("b", 40.0), hit("a", 1.0)]

    fused = reciprocal_rank_fusion([dense, lexical], k=60, top_k=2)

    assert fused[0].retrieval_score == pytest.approx(fused[1].retrieval_score)
    assert ids(fused) == ["a", "b"]  # a tie keeps the order of first mention


def test_fusion_truncates_to_top_k() -> None:
    dense = [hit("a"), hit("b"), hit("c")]

    assert ids(reciprocal_rank_fusion([dense, []], k=60, top_k=2)) == ["a", "b"]


def test_fusion_keeps_text_and_metadata_and_drops_nothing_else() -> None:
    passage = ScoredPassage(
        id="a", text="vacation", metadata={"doc": "handbook"}, retrieval_score=0.9
    )

    [fused] = reciprocal_rank_fusion([[passage]], k=60, top_k=1)

    assert (fused.text, fused.metadata, fused.rerank_score) == (
        "vacation",
        {"doc": "handbook"},
        None,
    )
    assert passage.retrieval_score == 0.9  # the input is not modified


def test_no_ranking_gives_no_passage() -> None:
    assert reciprocal_rank_fusion([], k=60, top_k=5) == []
    assert reciprocal_rank_fusion([[], []], k=60, top_k=5) == []


def test_a_smaller_k_widens_the_gap_between_ranks() -> None:
    ranking = [hit("a"), hit("b")]

    def gap(k: int) -> float:
        first, second = reciprocal_rank_fusion([ranking], k=k, top_k=2)
        return first.retrieval_score - second.retrieval_score

    assert gap(1) > gap(60)
