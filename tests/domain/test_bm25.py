"""``Bm25Index``: ranked lexical search over passages held in memory."""

from __future__ import annotations

import pytest

from rag_load_test.domain.bm25 import Bm25Index, tokenize_words


def index_of(**passages: str) -> Bm25Index:
    index = Bm25Index(k1=1.2, b=0.75)
    for passage_id, text in passages.items():
        index.add(passage_id, text)
    return index


def test_tokenize_words_lower_cases_and_keeps_accented_words_whole() -> None:
    assert tokenize_words("Congés payés: 25 jours_ouvrés!") == [
        "congés",
        "payés",
        "25",
        "jours",
        "ouvrés",
    ]


def test_search_ranks_the_passage_sharing_more_query_words_first() -> None:
    index = index_of(
        vacation="employees accrue vacation days every month",
        expenses="expense receipts are required for every claim",
        both="vacation days and expense claims",
    )

    ranked = [passage_id for passage_id, _ in index.search("vacation days", top_k=3)]

    assert ranked[:2] == ["both", "vacation"] or ranked[:2] == ["vacation", "both"]
    assert "expenses" not in ranked  # shares no word with the query


def test_a_rare_word_weighs_more_than_a_common_one() -> None:
    index = index_of(
        a="policy policy policy",
        b="policy sabbatical",
        c="policy handbook",
        d="policy travel",
    )

    [(best, _), *_] = index.search("policy sabbatical", top_k=4)

    assert best == "b"


def test_a_shorter_passage_wins_at_equal_term_counts() -> None:
    index = index_of(
        short="vacation policy",
        long="vacation policy " + "and many other unrelated words " * 5,
    )

    assert [pid for pid, _ in index.search("vacation", top_k=2)] == ["short", "long"]


def test_search_truncates_to_top_k_and_scores_descend() -> None:
    index = index_of(a="x y", b="x", c="x y z", d="other")

    hits = index.search("x", top_k=2)

    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1] > 0


def test_adding_a_known_id_replaces_its_text() -> None:
    index = index_of(a="vacation days")
    index.add("a", "expense receipts")

    assert len(index) == 1
    assert index.search("vacation", top_k=5) == []
    assert [pid for pid, _ in index.search("expense", top_k=5)] == ["a"]


@pytest.mark.parametrize("query", ["", "unknown words", "!!!"])
def test_a_query_matching_nothing_returns_no_passage(query: str) -> None:
    assert index_of(a="vacation days").search(query, top_k=5) == []


def test_an_empty_index_answers_without_dividing_by_zero() -> None:
    assert Bm25Index(k1=1.2, b=0.75).search("vacation", top_k=5) == []


def test_b_zero_ignores_passage_length() -> None:
    index = Bm25Index(k1=1.2, b=0.0)
    index.add("short", "vacation")
    index.add("long", "vacation " + "filler " * 20)

    [(_, first), (_, second)] = index.search("vacation", top_k=2)

    assert first == pytest.approx(second)
