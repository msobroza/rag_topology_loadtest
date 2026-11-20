"""The word "passage" is left only where it has a reason to stay.

ADR 0014 made doclang's vector-store record the workflow's unit of retrieval
and took the word with the class it named: ``records`` in code, "vector-store
record" in prose. A word left behind says that a passage and a record differ,
which they do not. This searches every tracked file of the repository, in any
case of the word, and fails on a line that holds it outside what KEPT and
KEPT_FOLDERS list, each with its reason. It follows ``test_old_names.py``.

A file that keeps one thing among others is listed by that thing's pattern,
never whole: the pattern is taken out of each line before the word is searched.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from test_old_names import tracked_files

from repo_paths import REPO_ROOT

THIS_FILE = "query_processing/tests/structure/test_the_word_passage.py"
WORD = re.compile("passage", re.IGNORECASE)
KEPT_FOLDERS = {
    "docs/adr/": "dated records, never reworded: ADR 0014 names the word as what goes",
    "docs/specs/": "dated records, never reworded, this build's spec included",
    "docs/plans/": "dated records, never reworded",
    "experiments/": "the registry of measurements, in the words of their day",
    "experiment_charts/": "the charts package tells measurements in the words "
    "of their day",
    "experiments-explained/": "the tour the charts package draws, in the words "
    "of its measurements",
    THIS_FILE: "it names what it searches for",
}


class Kept(NamedTuple):
    """One thing that keeps the word, the files that hold it, and why."""

    reason: str
    pattern: re.Pattern[str]
    files: frozenset[str]


USE_CASE = "query_processing_rag_usecase/"
TOOLKIT_TESTS = "query_processing/tests/"
KEPT = (
    Kept(
        "a prompt's sentence to the LLM: a model reads it and answers by it",
        re.compile(
            r"numbered context passages|Cite passages as \[n\]"
            r"|numbered fixture passages only"
        ),
        frozenset(
            {
                f"{USE_CASE}prompts.yaml",
                f"{USE_CASE}tests/test_workflow_file.py",
                "README.md",
                f"{TOOLKIT_TESTS}helpers/fixture_prompts.yaml",
                f"{TOOLKIT_TESTS}application/test_prompt_rendering.py",
            }
        ),
    ),
    Kept(
        "a metadata field's description: it goes into a prompt a model reads",
        re.compile(
            r"The reference the passage comes from"
            r"|the handbook topic a passage belongs to"
        ),
        frozenset(
            {
                f"{USE_CASE}workflow.yaml",
                f"{USE_CASE}tests/test_workflow_file.py",
                "README.md",
                f"{TOOLKIT_TESTS}helpers/fixture_workflow.yaml",
                f"{TOOLKIT_TESTS}entrypoints/qp_serve/test_qp_serve_output_schema.py",
            }
        ),
    ),
    Kept(
        "the E5 models' own prefix in front of a text to store: the text is "
        "the model's",
        re.compile(r"""["']passage: ["']"""),
        frozenset(
            {
                f"{TOOLKIT_TESTS}deploy/test_entry_scripts_through_app_sh.py",
                f"{USE_CASE}tests/helpers/trained_router.py",
                f"{USE_CASE}tests/test_pipeline_artifact.py",
            }
        ),
    ),
    Kept(
        "an index's or a collection's name: indexes and snapshots that exist "
        "are found by it",
        re.compile(r"rag_passages"),
        frozenset(
            {
                f"{USE_CASE}workflow.yaml",
                f"{USE_CASE}.env.example",
                f"{USE_CASE}tests/test_workflow_file.py",
                "README.md",
                "emulation/machine/index_snapshot.sh",
                f"{TOOLKIT_TESTS}emulation/test_emulation_machine_scripts.py",
                f"{TOOLKIT_TESTS}structure/test_old_names.py",
            }
        ),
    ),
    Kept(
        "the key of the emulation's snapshot metadata: the cache bucket holds "
        "snapshots written with it",
        re.compile(r'"passages": '),
        frozenset(
            {
                "emulation/machine/index_snapshot.sh",
                f"{TOOLKIT_TESTS}emulation/test_emulation_machine_scripts.py",
            }
        ),
    ),
    Kept(
        "the key of the old shape of a response and of an answer record: the "
        "load-test client reads it as records, with no end date",
        re.compile(r'(?:``|["`]|\| )passages\b'),
        frozenset(
            {
                "query_processing/src/query_processing/loadtest/answer_record.py",
                "query_processing/src/query_processing/testing/pinned_answer_record.py",
                f"{TOOLKIT_TESTS}helpers/answer_record_helpers.py",
                f"{TOOLKIT_TESTS}loadtest/test_answer_records_compare_alike.py",
                f"{TOOLKIT_TESTS}loadtest/test_qp_record_answers.py",
                f"{TOOLKIT_TESTS}testing/test_pinned_answer_record.py",
                f"{USE_CASE}tests/answer_records/test_pinned_answer_records.py",
                "README.md",
            }
        ),
    ),
    Kept(
        "the calibration setting's old name: start-up refuses it, and the "
        "README tells how to move from it",
        re.compile(r"calibration_passages", re.IGNORECASE),
        frozenset(
            {
                "query_processing/src/query_processing/config/components/"
                "component_settings.py",
                f"{TOOLKIT_TESTS}config/components/test_component_settings.py",
                "README.md",
            }
        ),
    ),
    Kept(
        "the glossary's lines that name the word to avoid it",
        re.compile(r"^_Avoid_: passage\b", re.MULTILINE),
        frozenset({"CONTEXT.md"}),
    ),
)


def lines_with_the_word(name: str, text: str) -> list[str]:
    """Each line of ``text`` that holds the word once KEPT's patterns are out.

    Example::

        lines_with_the_word("CONTEXT.md", "a passage\\n")  # ["CONTEXT.md:1: a passage"]
    """
    kept = [entry.pattern for entry in KEPT if name in entry.files]
    return [
        f"{name}:{number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), start=1)
        if WORD.search(without_the_kept(line, kept))
    ]


def without_the_kept(line: str, kept: list[re.Pattern[str]]) -> str:
    """``line`` with every match of the ``kept`` patterns taken out."""
    for pattern in kept:
        line = pattern.sub("", line)
    return line


def text_of(name: str) -> str:
    try:
        return (REPO_ROOT / name).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""  # a binary file names nothing


def test_the_word_is_left_only_where_a_reason_keeps_it() -> None:
    searched = [n for n in tracked_files() if not n.startswith(tuple(KEPT_FOLDERS))]
    found = [
        hit for name in searched for hit in lines_with_the_word(name, text_of(name))
    ]

    assert len(searched) > 100, "too few files were searched: the listing is wrong"
    assert found == [], "the word outside the kept places:\n" + "\n".join(found)


def test_each_kept_file_still_holds_what_keeps_the_word() -> None:
    # A listing that no longer matches would keep the word where it has no reason.
    unused = [
        (entry.reason, name)
        for entry in KEPT
        for name in sorted(entry.files)
        if not entry.pattern.search(text_of(name))
    ]

    assert unused == []


def test_the_search_finds_the_word_in_any_case_and_names_the_line() -> None:
    text = "one\nThe Passages kept\nPASSAGE\nrag_passages\n"

    assert lines_with_the_word("x.md", text) == [
        "x.md:2: The Passages kept",
        "x.md:3: PASSAGE",
        "x.md:4: rag_passages",
    ]


def test_a_kept_pattern_does_not_hide_the_word_beside_it() -> None:
    line = "index rag_passages holds 3 passages"

    assert lines_with_the_word("README.md", line) == [f"README.md:1: {line}"]
