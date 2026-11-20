"""The one-way rule: a use case installs the toolkit, never the reverse, so no file
of the toolkit's source names a use case: not in an import, not as a path or a
default. Moved out of test_layering.py, which holds the import table."""

from __future__ import annotations

import re
from collections.abc import Mapping

from repo_paths import TOOLKIT_ROOT

# --- one way: a use case installs the toolkit, never the reverse --------------------

# A use case is a distribution of its own beside the toolkit (ADR 0005), named
# query_processing_<name>_usecase (its distribution with hyphens). No file of the
# toolkit's source names one: not in an import, not as a path or a default.
USE_CASE_NAME = re.compile(r"query[_-]processing[_-]\w+?[_-]usecase")


def files_naming_a_use_case(files: Mapping[str, str]) -> list[str]:
    """The paths of ``files`` (path to text) whose text names a use case.

    Example::

        files_naming_a_use_case({"a.py": "import query_processing_rag_usecase"})
        # ["a.py"]
    """
    return sorted(path for path, text in files.items() if USE_CASE_NAME.search(text))


def test_the_use_case_rule_on_literal_files() -> None:
    files = {
        "a.py": "from query_processing_rag_usecase import settings\n",
        "b.py": "import query_processing  # the toolkit's own name\n",
        "c.yaml": "dependencies: [query-processing-legal-usecase]\n",
        "d.py": "# a use case's directory holds workflow.yaml\n",
    }
    assert files_naming_a_use_case(files) == ["a.py", "c.yaml"]


def test_no_file_of_the_toolkits_source_names_a_use_case() -> None:
    source = TOOLKIT_ROOT / "src"
    files = {
        str(path.relative_to(source)): path.read_text(encoding="utf-8")
        for path in sorted(source.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert files, f"no file was read under {source}: the walk is wrong"
    assert files_naming_a_use_case(files) == []
