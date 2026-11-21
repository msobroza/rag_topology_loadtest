"""The word "passage" is left only where it has a reason to stay.

ADR 0014 made doclang's vector-store record the workflow's unit of retrieval
and took the word with the class it named: ``records`` in code, "vector-store
record" in prose. A word left behind says that a passage and a record differ,
which they do not. This searches every tracked file of the repository, in any
case of the word, and fails on a line that holds it outside what KEPT and
KEPT_FOLDERS list, each with its reason. It follows ``test_old_names.py``.

A file that keeps one thing among others is listed by that thing's pattern,
never whole: the pattern is taken out of each line before the word is searched.
A pattern is the kept thing as it is written (a whole sentence, a quoted key, a
name between its delimiters), so it cannot hide the word in a new sentence or
in an identifier that merely starts with a kept name. A tracked file's own name
is searched too.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import pytest
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
SETTINGS_MODULE = (
    "query_processing/src/query_processing/config/components/component_settings.py"
)
OLD_KEY = (
    "the key of the old shape of a response and of an answer record: the "
    "load-test client reads it as records, with no end date"
)
ANSWER_RECORD_FILES = frozenset(
    {
        "query_processing/src/query_processing/loadtest/answer_record.py",
        "query_processing/src/query_processing/testing/pinned_answer_record.py",
        f"{TOOLKIT_TESTS}helpers/answer_record_helpers.py",
        f"{TOOLKIT_TESTS}loadtest/test_answer_records_compare_alike.py",
        f"{TOOLKIT_TESTS}loadtest/test_qp_record_answers.py",
    }
)
# Each pattern is the kept thing as it is written, a whole sentence, a quoted
# key or a name between its delimiters, never a phrase that new prose could
# begin with or hold: what it takes out of a line is that thing alone.
KEPT = (
    Kept(
        "a prompt's sentence to the LLM: a model reads it and answers by it",
        re.compile(
            r"You answer questions using only the numbered context passages\."
            r"|Cite passages as \[n\]\. If the context is insufficient, say so\."
            r"|Answer from the numbered fixture passages only\."
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
            r"(?:description[:=] ?\"?|- source: |topic: )"
            r"(?:The reference the passage comes from\."
            r"|the handbook topic a passage belongs to)"
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
        # The name where a setting, a path or a command gives it, whole.
        re.compile(r"(?:(?<=[`/=\-\"])|\b(?:index:?|in|default|Label) )rag_passages\b"),
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
        re.compile(
            r'"passages": (?:int\(records\)|\{INDEXED_RECORDS\})'
            r'|its key "passages": the cache bucket'
        ),
        frozenset(
            {
                "emulation/machine/index_snapshot.sh",
                f"{TOOLKIT_TESTS}emulation/test_emulation_machine_scripts.py",
            }
        ),
    ),
    Kept(
        OLD_KEY,
        # The key in a JSON object, a list of keys or a check that it is gone.
        re.compile(r'"passages"(?:\??: \[|, "records"| in place of| not in answer)'),
        frozenset(
            {
                "query_processing/src/query_processing/loadtest/answer_record.py",
                f"{TOOLKIT_TESTS}helpers/answer_record_helpers.py",
                f"{TOOLKIT_TESTS}loadtest/test_answer_records_compare_alike.py",
                f"{TOOLKIT_TESTS}loadtest/test_qp_record_answers.py",
                f"{TOOLKIT_TESTS}testing/test_pinned_answer_record.py",
                f"{USE_CASE}tests/answer_records/test_pinned_answer_records.py",
            }
        ),
    ),
    Kept(
        OLD_KEY,
        # A docstring's: the records or the list under the key.
        re.compile(
            r"under ``passages``|^\s*``passages``, each with its ``id``", re.MULTILINE
        ),
        ANSWER_RECORD_FILES,
    ),
    Kept(
        OLD_KEY,
        re.compile(r'answer\["passages"\] = None if entries is None'),
        frozenset({f"{TOOLKIT_TESTS}helpers/answer_record_helpers.py"}),
    ),
    Kept(
        OLD_KEY,
        # The comparison's report names no field under the old key.
        re.compile(r'"\| passages\[" not in report'),
        frozenset({f"{TOOLKIT_TESTS}loadtest/test_answer_records_compare_alike.py"}),
    ),
    Kept(
        OLD_KEY,
        # The README's two sentences that describe the old form.
        re.compile(
            r"^written in an old form: the vector-store records under `passages`, "
            r"each with$|^response's body and for a record's file alike: "
            r"`passages` is read as$",
            re.MULTILINE,
        ),
        frozenset({"README.md"}),
    ),
    Kept(
        "the calibration setting's old name: start-up refuses it, and the "
        "README tells how to move from it",
        # The name quoted, as a key or as a variable's last part, whole.
        re.compile(
            r"(?:``|[`'\".]|__|got )calibration_passages(?:``|[`'\"=])",
            re.IGNORECASE,
        ),
        frozenset(
            {
                SETTINGS_MODULE,
                f"{TOOLKIT_TESTS}config/components/" "test_component_settings.py",
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


def names_with_the_word(names: list[str]) -> list[str]:
    """Each of ``names``, a tracked file's path, that holds the word.

    Example::

        names_with_the_word(["a/passages.py", "b.md"])  # ["a/passages.py: its name"]
    """
    return [f"{name}: its name" for name in names if WORD.search(name)]


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
    found = names_with_the_word(searched) + [
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


ANSWER_RECORD = "query_processing/src/query_processing/loadtest/answer_record.py"
COMPARE_ALIKE = f"{TOOLKIT_TESTS}loadtest/test_answer_records_compare_alike.py"


@pytest.mark.parametrize(
    ("name", "line"),
    [
        (
            "README.md",
            "The rerank stage sends the numbered context passages to the LLM.",
        ),
        ("README.md", "Cite passages as [n], then list them."),
        ("README.md", "`source` is the handbook topic a passage belongs to."),
        ("README.md", "The rag_passages are then reranked."),
        ("emulation/machine/index_snapshot.sh", 'summary = {"passages": len(kept)}'),
        (ANSWER_RECORD, 'body["passages"] = kept'),
        (ANSWER_RECORD, "Each of the ``passages`` is scored."),
        ("README.md", "| `rerank` | `top_k` | `10` | passages kept after reranking |"),
        ("README.md", "| passages per request | 20 |"),
        ("README.md", "`/query` returns the answer and its `passages`."),
        ("README.md", 'The response holds "passages", each with an id.'),
        ("README.md", 'the list is "passages of the handbook"'),
        ("README.md", "`passages are returned in order`"),
        ("README.md", "the records under `passages` are scored"),
        (SETTINGS_MODULE, "calibration_passages: int = 20"),
    ],
)
def test_a_kept_phrase_in_a_new_sentence_is_found(name: str, line: str) -> None:
    # A kept thing is kept as it is written, not as a phrase that new prose
    # could begin with or hold.
    assert lines_with_the_word(name, line) == [f"{name}:1: {line}"]


@pytest.mark.parametrize(
    ("name", "line"),
    [
        (f"{USE_CASE}tests/test_workflow_file.py", "rag_passages_kept = 3"),
        ("README.md", "set `local.calibration_passages_old`"),
        ("README.md", "index: rag_passages_v2"),
    ],
)
def test_an_identifier_that_starts_with_a_kept_name_is_found(
    name: str, line: str
) -> None:
    assert lines_with_the_word(name, line) == [f"{name}:1: {line}"]


def test_a_table_cell_that_begins_with_the_word_is_kept_for_one_test_file() -> None:
    line = 'assert "| passages[" not in report'

    assert lines_with_the_word(COMPARE_ALIKE, line) == []
    assert lines_with_the_word(ANSWER_RECORD, line) == [f"{ANSWER_RECORD}:1: {line}"]
    assert lines_with_the_word("README.md", line) == [f"README.md:1: {line}"]


def test_the_readme_keeps_the_old_key_in_its_two_sentences_alone() -> None:
    old_key = [
        e.pattern for e in KEPT if e.reason == OLD_KEY and "README.md" in e.files
    ]
    kept = [
        line
        for line in text_of("README.md").splitlines()
        if WORD.search(line) and any(pattern.search(line) for pattern in old_key)
    ]

    assert kept == [
        "written in an old form: the vector-store records under `passages`, each with",
        "response's body and for a record's file alike: `passages` is read as",
    ]


def test_a_file_whose_name_holds_the_word_is_found() -> None:
    names = ["src/domain/retrieval/passages.py", "docs/Passage.md", "README.md"]

    assert names_with_the_word(names) == [
        "src/domain/retrieval/passages.py: its name",
        "docs/Passage.md: its name",
    ]
