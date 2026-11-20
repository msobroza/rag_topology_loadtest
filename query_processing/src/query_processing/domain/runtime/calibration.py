"""Calibration: the candidates a CPU-bound stage is timed under, and the fastest.

At start-up a CPU-bound stage (embed or rerank on an in-process model) is timed
under each candidate batch size and thread count left at 0 by the settings, and
keeps the fastest. This module holds the rules, no clock and no model: which
candidates (``halvings``), which trial wins (``fastest``), what a calibration
reports (``CalibrationReport``) and the request every candidate serves
(``calibration_sample``). A model server (OVMS) is calibrated the same way on
its stream and thread counts. The loops are
``infrastructure/local_models/stage_calibration.py`` (in-process) and
``infrastructure/calibration/served_stage_calibration.py`` (model server).

Example:
    >>> halvings(20)
    (20, 10, 5, 2, 1)
    >>> slow, quick = StageTuning(2, 20), StageTuning(2, 2)
    >>> fastest([TimedTrial(slow, 5.5), TimedTrial(quick, 2.9)])
    StageTuning(threads=2, batch_size=2, streams=1)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..retrieval.corpus import synthetic_corpus, synthetic_questions


@dataclass(frozen=True)
class StageTuning:
    """The parameters calibration chooses for a stage.

    Example:
        >>> StageTuning(threads=2, batch_size=20, streams=2)  # a model server's
        StageTuning(threads=2, batch_size=20, streams=2)
    """

    threads: int  # native threads one call of the stage may use (all its streams')
    batch_size: int  # texts (or passages) per model batch
    # Requests the stage's model runs at once: 1 for an in-process stage, which
    # runs one call at a time; a model server splits its threads among them.
    streams: int = 1


@dataclass(frozen=True)
class TimedTrial:
    """A tuning and the seconds one request took under it (the fewest of the rounds)."""

    tuning: StageTuning
    seconds: float


@dataclass(frozen=True)
class CalibrationReport:
    """What one calibration of a stage found: its choice and every trial.

    Example:
        >>> trial = TimedTrial(StageTuning(2, 2), 2.94)
        >>> report = CalibrationReport("rerank", StageTuning(2, 2), (trial,),
        ...                            seconds=68.4, budget_s=90.0, cut_short=False)
        >>> report.chosen_seconds
        2.94
    """

    stage: str  # the stage's name: embed, rerank
    chosen: StageTuning
    trials: tuple[TimedTrial, ...]  # in the order tried; none when nothing was timed
    seconds: float  # spent calibrating, loads included; 0 when nothing was timed
    budget_s: float
    cut_short: bool  # the budget stopped the search before every candidate ran
    # Requests each trial served at once: 1 for an in-process stage; a model
    # server's stream count is chosen for this load.
    requests: int = 1

    @property
    def chosen_seconds(self) -> float | None:
        """The seconds a trial took under the choice; None when none was timed.

        A trial is ``requests`` requests at once, so this is the time until the
        last of them was answered.

        Returns:
            The fewest seconds of the choice's trials, or None.
        """
        timed = [trial.seconds for trial in self.trials if trial.tuning == self.chosen]
        return min(timed, default=None)


@dataclass(frozen=True)
class CalibrationSample:
    """The request every candidate serves: one question and its passages."""

    question: str
    records: tuple[str, ...]


def halvings(top: int) -> tuple[int, ...]:
    """Return ``top``, ``top // 2``, ... down to 1: the candidates of one parameter.

    Args:
        top: The largest candidate: the whole request, or the thread budget's share.

    Returns:
        The distinct candidates, largest first.

    Raises:
        ValueError: If ``top`` is below 1.

    Example:
        >>> halvings(20)
        (20, 10, 5, 2, 1)
    """
    if top < 1:
        raise ValueError(f"halvings need top >= 1, got top={top}")
    candidates = [top]
    while candidates[-1] > 1:
        candidates.append(candidates[-1] // 2)
    return tuple(candidates)


def fastest(trials: Sequence[TimedTrial]) -> StageTuning:
    """Return the tuning of the trial with the fewest seconds.

    Args:
        trials: The trials, in the order they were tried.

    Returns:
        The fastest tuning; on a tie, the one tried first.

    Raises:
        ValueError: If there is no trial.

    Example:
        >>> first, second = StageTuning(2, 20), StageTuning(1, 20)
        >>> fastest([TimedTrial(first, 3.0), TimedTrial(second, 3.0)])
        StageTuning(threads=2, batch_size=20, streams=1)
    """
    if not trials:
        raise ValueError("fastest needs at least one trial, got no trial")
    # min keeps the first of equal keys, so a tie goes to the tuning tried first.
    return min(trials, key=lambda trial: trial.seconds).tuning


def calibration_sample(
    records: int, *, shortest: int = 150, longest: int = 800, seed: int = 1234
) -> CalibrationSample:
    """Build a deterministic question and ``records`` passages of spread lengths.

    The passages are the synthetic corpus's texts, repeated and cut to lengths
    spread evenly from ``shortest`` to ``longest`` characters, then shuffled.
    Spread, because a batch is padded to its longest passage and that padding is
    what batch size trades against; real candidates vary that much (137 to 816
    characters on the corpus of the GCP sitting of 2026-11-15). Shuffled, because
    retrieval returns them by score, not by length.

    Args:
        records: How many passages: the request's usual size.
        shortest: Characters of the shortest passage.
        longest: Characters of the longest passage.
        seed: Seed of the corpus and of the shuffle.

    Returns:
        The question and the passages.

    Raises:
        ValueError: If ``records`` is below 1.

    Example:
        >>> sample = calibration_sample(20)
        >>> len(sample.records)
        20
    """
    if records < 1:
        raise ValueError(f"a calibration sample needs records >= 1, got {records=}")
    texts = [doc.text for doc in synthetic_corpus(records, seed=seed)]
    steps = max(records - 1, 1)
    lengths = [shortest + (longest - shortest) * i // steps for i in range(records)]
    pairs = zip(texts, lengths, strict=True)
    cut = [_repeated_to(text, length) for text, length in pairs]
    random.Random(seed).shuffle(cut)
    return CalibrationSample(synthetic_questions(1, seed=seed)[0], tuple(cut))


def _repeated_to(text: str, length: int) -> str:
    # Repeat a passage until it is long enough, then cut it to exactly ``length``.
    repeats = length // len(text) + 1
    return " ".join([text] * repeats)[:length]
