"""The pass-through of a model server's flags, which OVMS and vLLM share.

A server takes named settings and a pass-through variable of raw flags; a flag
given both ways is refused, under either spelling (``_`` or ``-``, with or
without ``=value``), so it never reaches the server twice.
"""

from __future__ import annotations

import pytest

from query_processing.config.apps.flag_pass_through import (
    names_flag,
    refuse_given_twice,
    split_pass_through,
)


@pytest.mark.parametrize(
    "word", ["--rest_workers", "--rest-workers", "--rest_workers=8", "--rest-workers=8"]
)
def test_a_flag_is_recognised_under_each_spelling(word: str) -> None:
    assert names_flag(word, "--rest_workers")


@pytest.mark.parametrize("word", ["--max_num_seqs=8", "--max-num-seqs"])
def test_a_hyphenated_flag_is_recognised_under_each_spelling(word: str) -> None:
    assert names_flag(word, "--max-num-seqs")


@pytest.mark.parametrize("word", ["--rest_workers_extra", "--rest_worker", "8"])
def test_another_flag_with_the_same_start_is_not_it(word: str) -> None:
    assert not names_flag(word, "--rest_workers")


def test_the_words_are_split_as_a_shell_would() -> None:
    words = split_pass_through(
        "--log_level 'DEBUG' --x=\"a b\"", variable="QP_OVMS_ARGS"
    )
    assert words == ["--log_level", "DEBUG", "--x=a b"]


def test_an_empty_pass_through_gives_no_word() -> None:
    assert split_pass_through("", variable="QP_OVMS_ARGS") == []


def test_an_unbalanced_quote_names_the_variable() -> None:
    with pytest.raises(ValueError, match=r"QP_OVMS_ARGS.*--log_level 'DEBUG"):
        split_pass_through("--log_level 'DEBUG", variable="QP_OVMS_ARGS")


@pytest.mark.parametrize(
    "words",
    [
        ["--rest-workers=8"],
        ["--rest_workers", "8"],
        ["--log_level", "INFO", "--rest-workers=8"],
    ],
)
def test_a_flag_given_both_ways_is_refused_naming_both(words: list[str]) -> None:
    with pytest.raises(
        ValueError, match=r"QP_OVMS_REST_WORKERS=4.*QP_OVMS_ARGS.*8.*state it once"
    ):
        refuse_given_twice(
            {"--rest_workers": "QP_OVMS_REST_WORKERS=4"},
            words,
            pass_through="QP_OVMS_ARGS",
        )


def test_flags_given_one_way_each_pass() -> None:
    refuse_given_twice(
        {"--rest_workers": "QP_OVMS_REST_WORKERS=4"},
        ["--log_level", "INFO", "--rest_workers_extra=1"],
        pass_through="QP_OVMS_ARGS",
    )
