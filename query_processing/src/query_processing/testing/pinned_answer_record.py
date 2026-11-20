"""The pinned answer record: what the workflow app answers on fake backends, committed.

A distribution's tests record its workflow app in process, every component on
its fake backend, with ``qp-record-answers``, and compare the record with the
one committed beside the test with ``qp-compare-answers``: every answer
identical, its text included (ROUGE-L at 1). Each answer's text is then held
exactly, so a change of punctuation, case, line breaks or a ``[1]`` marker,
which ROUGE-L does not see, fails too. The committed record is the
"before" a change to how the graph runs is held to. The committed records are
of the new form, the vector-store records under ``records``; one of the old
form, its records under ``passages``, would be read through the answer
record's mapping into the new form, so it still compares with a record made
today. A change that alters the answers on purpose makes the record
again with the record's pin command, which runs the same test with
``--pin-answers``; the failing test names it.

The app is started with ``TestClient``, an ``httpx.Client`` that
``qp-record-answers`` is handed, so no socket is opened. The load test's
variables are cleared and both commands run in a folder of the test's, where
no ``.env`` is, so a record holds only what the test names. A ``conftest.py``
adds the option with ``add_pin_answers_option`` and reads it with
``pin_answers_asked``.

Example:
    >>> pinned = PinnedAnswerRecord(
    ...     Path("tests/testing/answer_records/fixture_workflow"),
    ...     pin_command="make -C query_processing pin-answers",
    ...     variables={"QP_LOADTEST_QUESTIONS": "10"},
    ... )
    >>> with workflow_client(settings) as client:
    ...     check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=False)
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from fastapi.testclient import TestClient

from ..composition.composition_root import CompositionRoot
from ..config.workflow_settings import WorkflowSettings
from ..entrypoints.qp_serve import create_app
from ..loadtest.answer_record import ANSWER_RECORD_SUFFIX, read_answer_record
from ..loadtest.loadtest_settings import LoadTestSettings
from ..loadtest.qp_compare_answers import compare_answers_main
from ..loadtest.qp_record_answers import record_answers_main

if TYPE_CHECKING:  # pytest is a dev dependency: a test's, never an App's
    import pytest

PIN_ANSWERS_OPTION = "--pin-answers"
# The fake LLM's answer is a function of its prompt, so its text is held too:
# ROUGE-L at 1 is the same words in the same order.
_TEXT_HELD_TOO = {
    "QP_LOADTEST_ANSWER_ROUGE_L": "true",
    "QP_LOADTEST_ANSWER_ROUGE_L_THRESHOLD": "1",
}
_RECORDED_NOW = "recorded-now"  # the run name of the record a test makes


class PinnedAnswersDifferError(AssertionError):
    """The app does not answer as its pinned record, or no record is pinned.

    An ``AssertionError``, so pytest reports it as a failed test; the message
    names the pin command.

    Example:
        >>> raise PinnedAnswersDifferError("... run `make pin-answers` ...")
    """


@dataclass(frozen=True)
class PinnedAnswerRecord:
    """A committed answer record, the command that pins it again, and its variables.

    Args:
        prefix: The ``--out`` prefix the record is written under; its last
            component names the pinned run in the comparison.
        pin_command: What makes the record again, named by a failing test.
        variables: The load test's variables the record is made with, such as
            ``QP_LOADTEST_QUESTIONS`` or ``QP_LOADTEST_QUESTIONS_FILE``.

    Example:
        >>> PinnedAnswerRecord(Path("answers/rag"), "make pin-answers").path.name
        'rag_answers.json'
    """

    prefix: Path
    pin_command: str
    variables: Mapping[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """The committed file, ``<prefix>_answers.json``."""
        return Path(f"{self.prefix}{ANSWER_RECORD_SUFFIX}")


def add_pin_answers_option(parser: pytest.Parser) -> None:
    """Add ``--pin-answers`` to pytest, from a ``conftest.py``'s ``pytest_addoption``.

    Args:
        parser: The parser pytest hands the hook.

    Example:
        >>> def pytest_addoption(parser):
        ...     add_pin_answers_option(parser)
    """
    parser.addoption(
        PIN_ANSWERS_OPTION,
        action="store_true",
        default=False,
        help="make each pinned answer record again instead of comparing with it",
    )


def pin_answers_asked(config: pytest.Config) -> bool:
    """Return whether pytest was given ``--pin-answers``.

    Args:
        config: The run's configuration, ``request.config`` in a fixture.

    Returns:
        True when the pinned records are to be made again.

    Example:
        >>> pin_answers_asked(request.config)
        False
    """
    return bool(config.getoption(PIN_ANSWERS_OPTION))


def clear_loadtest_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every variable ``LoadTestSettings`` reads, for this test only.

    Args:
        monkeypatch: The test's, which sets them back after it.

    Example:
        >>> clear_loadtest_variables(monkeypatch)
    """
    for name in LoadTestSettings.model_fields:
        monkeypatch.delenv(LoadTestSettings.variable_of(name), raising=False)


def use_loadtest_variables(
    monkeypatch: pytest.MonkeyPatch, workdir: Path, variables: Mapping[str, str]
) -> None:
    """Run in ``workdir`` (no ``.env``) with exactly these load-test variables.

    Args:
        monkeypatch: The test's, which undoes both after it.
        workdir: A folder of the test's, the working directory from now on.
        variables: The load test's variables to set, by name.

    Example:
        >>> use_loadtest_variables(
        ...     monkeypatch, tmp_path, {"QP_LOADTEST_QUESTIONS": "3"}
        ... )
    """
    clear_loadtest_variables(monkeypatch)
    monkeypatch.chdir(workdir)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


@contextmanager
def workflow_client(
    settings: WorkflowSettings, transport: httpx.AsyncBaseTransport | None = None
) -> Generator[TestClient, None, None]:
    """The workflow app on ``settings``, started; its calls out go to ``transport``.

    Args:
        settings: The workflow settings the app is built from.
        transport: Where a served component's calls go; None for the network.

    Yields:
        A client of the app, in process.

    Example:
        >>> with workflow_client(settings) as client:
        ...     client.get("/readyz").status_code
        200
    """
    composition = CompositionRoot(settings, transport=transport)
    with TestClient(create_app(composition)) as client:
        yield client


def record_answers_of(client: httpx.Client, prefix: Path) -> int:
    """Run ``qp-record-answers`` against ``client``'s app; return its status.

    Args:
        client: The app's client, which the command does its HTTP with.
        prefix: The ``--out`` prefix of the record.

    Returns:
        The command's exit status, 0 once the record is written.

    Example:
        >>> record_answers_of(client, tmp_path / "before")
        0
    """
    argv = ["--host", str(client.base_url), "--out", str(prefix)]
    return record_answers_main(argv, client=client)


def compare_answer_records(*prefixes: Path, out: Path) -> int:
    """Run ``qp-compare-answers`` over ``prefixes`` into ``out``; return its status.

    Args:
        *prefixes: The records' prefixes, the baseline first.
        out: The markdown report written.

    Returns:
        The command's exit status: 0 identical, 1 different, 2 not comparable.

    Example:
        >>> compare_answer_records(tmp_path / "a", tmp_path / "b", out=report)
        0
    """
    return compare_answers_main([*map(str, prefixes), "--out", str(out)])


def check_pinned_answers(
    monkeypatch: pytest.MonkeyPatch,
    client: httpx.Client,
    pinned: PinnedAnswerRecord,
    workdir: Path,
    *,
    pin: bool,
) -> None:
    """Record ``client``'s app, then pin the record or compare it with the pinned one.

    Args:
        monkeypatch: The test's: the load test's variables are set with it.
        client: The workflow app's client, in process.
        pinned: The committed record and how it is made.
        workdir: A folder of the test's for the new record and the report.
        pin: True writes the record to ``pinned.path``, the pin command's run.

    Raises:
        PinnedAnswersDifferError: If the app's answers differ from the pinned
            record, or no record is pinned there; the message holds the
            report and names the pin command.

    Example:
        >>> check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=False)
    """
    use_loadtest_variables(monkeypatch, workdir, {**_TEXT_HELD_TOO, **pinned.variables})
    if pin:
        pinned.prefix.parent.mkdir(parents=True, exist_ok=True)
        _recorded(client, pinned.prefix)
        return
    if not pinned.path.is_file():
        raise PinnedAnswersDifferError(
            f"no answer record pinned at {pinned.path}; {_pin_hint(pinned)}"
        )
    report = workdir / "pinned-answers-comparison.md"
    status = compare_answer_records(
        pinned.prefix, _recorded(client, workdir / _RECORDED_NOW), out=report
    )
    if status != 0:
        found = report.read_text(encoding="utf-8") if report.is_file() else ""
        raise PinnedAnswersDifferError(
            f"the app does not answer as {pinned.path} (qp-compare-answers exit "
            f"{status}, its reason on stderr when 2):\n{found}\n{_pin_hint(pinned)}"
        )
    _check_texts_equal(pinned, workdir / _RECORDED_NOW)


def _check_texts_equal(pinned: PinnedAnswerRecord, recorded: Path) -> None:
    # ROUGE-L at 1 lets a change of punctuation, case, layout or a [1] marker
    # pass; the fakes answer the same text every run, so equality holds here.
    before = read_answer_record(pinned.prefix).answers
    now = read_answer_record(recorded).answers
    differing = [
        f"answer {index} ({then.endpoint}): {then.answer!r} is now {it.answer!r}"
        for index, (then, it) in enumerate(zip(before, now, strict=True))
        if then.answer != it.answer
    ]
    if differing:
        raise PinnedAnswersDifferError(
            f"the app does not answer as {pinned.path}: the text of "
            f"{len(differing)} answer(s) differs, the words being the same:\n"
            + "\n".join(differing)
            + f"\n{_pin_hint(pinned)}"
        )


def _recorded(client: httpx.Client, prefix: Path) -> Path:
    # A record that cannot be made says why on stderr; nothing is compared.
    status = record_answers_of(client, prefix)
    if status != 0:
        raise PinnedAnswersDifferError(
            f"qp-record-answers exit {status} for {prefix}: its reason on stderr"
        )
    return prefix


def _pin_hint(pinned: PinnedAnswerRecord) -> str:
    return (
        f"if the change alters the answers on purpose, run `{pinned.pin_command}` "
        f"from the repository's root and commit {pinned.path.name}"
    )
