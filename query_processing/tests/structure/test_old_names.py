"""No old name of the toolkit is left: not its package, a command or its prefix.

The toolkit took new names with no alias for the old ones (``query_processing``,
``qp-*``, ``QP_``), so a name left behind is a bug nothing else reports: a
variable under the old prefix is ignored, not refused, and an old command is not
found. This searches every tracked file of the repository and fails on a hit
outside the places ALLOWED lists, each with its reason, and the README's one
marked section about moving from the old names.

The emulation keeps the names it has outside its machines (its Terraform
resources and their name, the cache bucket, labels, the snapshot repository,
the state's prefix: ``rag-emulation``, ``rag-emu``), since renaming them would
orphan what the bucket and the state hold. None of them is an old name of the
toolkit, so the search does not match them and they need no entry here.
"""

from __future__ import annotations

import re
import subprocess

from repo_paths import REPO_ROOT

THIS_FILE = "query_processing/tests/structure/test_old_names.py"
ALLOWED = {
    "docs/": "dated records (the ADRs, notes of past sittings) keep the names of "
    "their day and are never rewritten",
    "experiments/": "the registry of measurements: a sitting's record holds the "
    "settings and commands as they were named the day it ran, and is never "
    "rewritten",
    THIS_FILE: "it names what it searches for",
}
# The README's section about moving from the old names, which has to name them.
README_BEGIN = "<!-- old-names:begin"
README_END = "<!-- old-names:end -->"
OLD_COMMANDS = (
    "serve|ingest|compare|model-server|served-models|export-openvino|calibrate"
    "|ovms-config|fetch-corpus|deploy|fake-domino|domino-emulation"
)
OLD_NAMES = re.compile(
    r"rag_load_test|rag-load-test"  # the package and its distribution
    rf"|(?<![\w-])rag-(?:{OLD_COMMANDS})"  # a command, .rag-deploy-cache.json too
    rf"|(?<![A-Za-z0-9])rag_(?:{OLD_COMMANDS.replace('-', '_')})"  # its module
    r"|RagSettings|rag_settings"  # the workflow settings
    r"|(?:(?<![\w-])|(?<=\\n))RAG_\w*"  # a variable, after a \n escape too
    r"|(?<![\w-])rag_\w*?__\w*"  # a variable's path in lower case
)
# Names that start like a variable but are constants named after the RAG.
NOT_A_VARIABLE = {
    "RAG_STAGE_TYPES",  # test_stage_types.py: the RAG's four stage types
    "RAG_SYSTEM_COMMAND_HOME",  # test_placement.py: the RAG system's commands
}


def tracked_files() -> list[str]:
    """Every file git tracks in the repository, by its path from the root."""
    listed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return [name for name in listed.stdout.splitlines() if (REPO_ROOT / name).is_file()]


def text_of(name: str) -> str:
    try:
        return (REPO_ROOT / name).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""  # a binary file names nothing


def without_the_readme_section(text: str) -> str:
    """The README's text, its section about moving from the old names left out."""
    begin, end = text.index(README_BEGIN), text.index(README_END)
    return text[:begin] + text[end:]


def old_names_in(text: str) -> list[str]:
    """Each old name ``text`` holds, in order.

    Example::

        old_names_in("exec rag-serve with RAG_ROLE")  # ["rag-serve", "RAG_ROLE"]
    """
    found = (match.group(0) for match in OLD_NAMES.finditer(text))
    return [name for name in found if name not in NOT_A_VARIABLE]


def searched_text(name: str) -> str:
    text = text_of(name)
    return without_the_readme_section(text) if name == "README.md" else text


def test_no_old_name_is_left_outside_the_allowed_places() -> None:
    searched = [n for n in tracked_files() if not n.startswith(tuple(ALLOWED))]
    hits = {name: sorted(set(old_names_in(searched_text(name)))) for name in searched}

    assert len(searched) > 100, "too few files were searched: the listing is wrong"
    assert {name: found for name, found in hits.items() if found} == {}


def test_the_readme_has_one_marked_section_that_moves_from_the_old_names() -> None:
    readme = text_of("README.md")
    assert (readme.count(README_BEGIN), readme.count(README_END)) == (1, 1)
    section = readme[readme.index(README_BEGIN) : readme.index(README_END)]

    # The first deploy creates new Apps; the old ones stay until deleted by hand.
    assert "creates new Apps" in section
    assert "`qp-rag-workflow`" in section
    assert "deleted by hand" in section
    assert {"rag_load_test", "rag-serve", "RAG_COMPONENTS__LLM__API__API_KEY"} <= set(
        old_names_in(section)
    )
    # A filter or a prefilter on the handbook's old key finds nothing, and says
    # nothing: the section's table gives the key that replaces it.
    rows = [row for row in section.splitlines() if row.startswith("| the handbook")]
    assert any("`chunk`" in row and "`sequence`" in row for row in rows), rows


def test_the_search_finds_each_kind_of_old_name() -> None:
    text = (
        "import rag_load_test; pip install rag-load-test; exec rag-serve; "
        "rag-domino-emulation; .rag-deploy-cache.json; rag_calibrate.py; "
        "rag-ovms-config in rag_ovms_config.py; "
        "test_rag_deploy.py; RagSettings in rag_settings.py; ${RAG_ROLE:-}; "
        '"# a\\nRAG_ES_URL=x"; rag_graph__stages__rerank__top_k'
    )

    assert old_names_in(text) == [
        "rag_load_test",
        "rag-load-test",
        "rag-serve",
        "rag-domino-emulation",
        "rag-deploy",
        "rag_calibrate",
        "rag-ovms-config",
        "rag_ovms_config",
        "rag_deploy",
        "RagSettings",
        "rag_settings",
        "RAG_ROLE",
        "RAG_ES_URL",
        "rag_graph__stages__rerank__top_k",
    ]


def test_the_search_leaves_the_names_that_stay() -> None:
    # The wire contract, the use case's own values, the emulation's names
    # outside its machines, App names under a prefix, and the RAG's identifiers.
    text = (
        "X-RAG-Topology X-RAG-Run-Label rag_passages rag-emulation rag-emu "
        "rag-reranker qp-rag-workflow plan_rag_request RagDependencies "
        "RAG_STAGE_TYPES query_processing_rag_usecase rag_topology_loadtest "
        "QP_ROLE qp-serve query_processing"
    )

    assert old_names_in(text) == []
