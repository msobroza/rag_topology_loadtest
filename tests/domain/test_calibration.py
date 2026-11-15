"""``domain.calibration``: the candidates, the fastest trial and the sample request."""

from __future__ import annotations

import pytest

from rag_load_test.domain.calibration import (
    StageTuning,
    TimedTrial,
    calibration_sample,
    fastest,
    halvings,
)

# --- halvings ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("top", "expected"),
    [
        (20, (20, 10, 5, 2, 1)),
        (8, (8, 4, 2, 1)),
        (3, (3, 1)),
        (1, (1,)),
    ],
)
def test_halvings_go_from_top_down_to_one(top: int, expected: tuple[int, ...]) -> None:
    assert halvings(top) == expected


@pytest.mark.parametrize("top", [0, -3])
def test_halvings_refuse_a_top_below_one(top: int) -> None:
    with pytest.raises(ValueError, match=f"top={top}"):
        halvings(top)


# --- fastest -------------------------------------------------------------------------


def trial(threads: int, batch_size: int, seconds: float) -> TimedTrial:
    return TimedTrial(StageTuning(threads=threads, batch_size=batch_size), seconds)


def test_fastest_is_the_tuning_with_the_fewest_seconds() -> None:
    trials = [trial(2, 20, 5.5), trial(2, 2, 3.4), trial(1, 2, 4.0)]

    assert fastest(trials) == StageTuning(threads=2, batch_size=2)


def test_a_tie_keeps_the_tuning_tried_first() -> None:
    trials = [trial(2, 20, 3.0), trial(2, 10, 3.0), trial(1, 10, 3.0)]

    assert fastest(trials) == StageTuning(threads=2, batch_size=20)


def test_fastest_of_no_trial_is_an_error() -> None:
    with pytest.raises(ValueError, match="no trial"):
        fastest([])


# --- the sample request --------------------------------------------------------------


def test_the_sample_is_deterministic() -> None:
    assert calibration_sample(20) == calibration_sample(20)


@pytest.mark.parametrize("passages", [1, 2, 20, 50])
def test_the_sample_has_the_passages_asked(passages: int) -> None:
    sample = calibration_sample(passages)

    assert len(sample.passages) == passages
    assert sample.question.endswith("?")


def test_the_sample_passages_spread_from_short_to_long() -> None:
    lengths = sorted(len(p) for p in calibration_sample(20).passages)

    assert 100 <= lengths[0] <= 200
    assert 700 <= lengths[-1] <= 800
    assert len(set(lengths)) == 20  # spread, not a few sizes repeated


def test_the_sample_passages_are_not_in_length_order() -> None:
    # Retrieval returns candidates in score order, not length order.
    lengths = [len(p) for p in calibration_sample(20).passages]

    assert lengths not in (sorted(lengths), sorted(lengths, reverse=True))


def test_the_sample_refuses_no_passage() -> None:
    with pytest.raises(ValueError, match="passages=0"):
        calibration_sample(0)
