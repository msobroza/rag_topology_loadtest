"""``qp-compare-answers``: answer records in, what differs from the first as markdown.

Each prefix names ``<prefix>_answers.json``, written by ``qp-record-answers``;
a run is named by the last component of its prefix, and the first is the
baseline, as ``qp-compare`` does for statistics. The report lists the runs
with the topology alias and run label each saw, one row per run (requests,
identical, different), then every answer that differs from the baseline's:
its endpoint, its question, the field, the baseline's value and the run's.
What is compared, and how, is ``answer_comparison``'s; the score tolerance
and whether the answer's text is compared are settings of the load test
(``QP_LOADTEST_SCORE_TOLERANCE``, ``QP_LOADTEST_COMPARE_ANSWER_TEXT``).

The exit status is 0 when every run answers as the baseline does, 1 when an
answer differs, and 2 when the records cannot be compared: a file that is
not an answer record, two prefixes of one last component, or records of
different requests, refused before any comparison.

Example:
    $ qp-compare-answers results/monolith results/split-all --out answers.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .answer_comparison import (
    AnswerDifference,
    AnswerTolerance,
    answer_differences,
    different_answers,
    require_same_requests,
)
from .answer_record import AnswerRecord, read_answer_record
from .loadtest_settings import LoadTestSettings
from .target_record import TargetSighting

ANSWERS_DIFFER = 1
NOT_COMPARABLE = 2  # as argparse exits on a usage error


def render_answer_comparison(
    records: Mapping[str, AnswerRecord], tolerance: AnswerTolerance
) -> tuple[str, bool]:
    """Render the comparison of each record against the first.

    Args:
        records: Each run's record by run name, the baseline first; all of
            the same requests.
        tolerance: What two answers may differ by.

    Returns:
        The markdown, and whether any answer differs.

    Example:
        >>> render_answer_comparison({"a": record, "b": record}, tolerance)[1]
        False
    """
    names = list(records)
    baseline = records[names[0]]
    found = {
        name: answer_differences(baseline, records[name], tolerance)
        for name in names[1:]
    }
    lines = ["# Answer comparison", "", *_intro_lines(records, tolerance)]
    lines += [
        "| run | requests | identical | different |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [_count_row(name, records[name], found.get(name, [])) for name in names]
    lines.append("")
    for name, differences in found.items():
        lines += _difference_lines(name, names[0], differences)
    return "\n".join(lines), any(found.values())


def _intro_lines(
    records: Mapping[str, AnswerRecord], tolerance: AnswerTolerance
) -> list[str]:
    titles = []
    for name, record in records.items():
        pairs = "; ".join(map(_described, record.seen))
        titles.append(f"{name} ({pairs})" if pairs else name)
    text = "compared" if tolerance.compare_answer_text else "not compared"
    return [
        f"Runs: {', '.join(titles)}. Each is compared with `{next(iter(records))}`; "
        f"scores within {tolerance.score_tolerance:g}, the answer's text {text}.",
        "",
    ]


def _described(sighting: TargetSighting) -> str:
    label = f"run label `{sighting.run_label}`"
    return f"topology `{sighting.topology}`, {label}" if sighting.topology else label


def _count_row(
    name: str, record: AnswerRecord, differences: list[AnswerDifference]
) -> str:
    total, different = len(record.answers), different_answers(differences)
    return f"| {name} | {total} | {total - different} | {different} |"


def _difference_lines(
    name: str, baseline: str, differences: list[AnswerDifference]
) -> list[str]:
    if not differences:
        return []
    header = f"| request | endpoint | question | field | {baseline} | {name} |"
    rows = [
        f"| {d.request} | {d.endpoint} | {_cell(d.question)} | {d.field} "
        f"| {_cell(d.baseline)} | {_cell(d.run)} |"
        for d in differences
    ]
    return [
        f"## {name}: what differs",
        "",
        header,
        "| ---: | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]


def _cell(value: object) -> str:
    # A table cell holds one line, and a | would end it.
    return str(value).replace("|", "\\|").replace("\n", " ")


def compare_answers_main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-compare-answers <prefix> <prefix>... [--out <file>]``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        0 when every run answers as the baseline, 1 when an answer differs,
        2 when the records cannot be compared (the reason on stderr).
    """
    args = _parser().parse_args(argv)
    settings = LoadTestSettings()
    tolerance = AnswerTolerance(settings.score_tolerance, settings.compare_answer_text)
    try:
        records = _comparable_records(args.prefixes)
    except ValueError as exc:
        print(f"qp-compare-answers: {exc}", file=sys.stderr)
        return NOT_COMPARABLE
    markdown, differ = render_answer_comparison(records, tolerance)
    if args.out is None:
        print(markdown)
    else:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return ANSWERS_DIFFER if differ else 0


def _comparable_records(prefixes: Sequence[Path]) -> dict[str, AnswerRecord]:
    # Every record read, then each held to the baseline's requests. Two
    # prefixes of one last component would be one run, the other dropped.
    names = [prefix.name for prefix in prefixes]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(
            f"two runs named {twice[0]!r}: a run is named by the last component "
            "of its prefix; expected prefixes whose last components differ"
        )
    records = {prefix.name: read_answer_record(prefix) for prefix in prefixes}
    named = list(records.items())
    for run in named[1:]:
        require_same_requests(named[0], run)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qp-compare-answers",
        description="Compare answer records against the first.",
    )
    parser.add_argument(
        "prefixes", nargs="+", type=Path, help="qp-record-answers --out prefixes"
    )
    parser.add_argument(
        "--out", type=Path, help="write markdown here (default: stdout)"
    )
    return parser
