"""Settings named by their environment path, for the tests.

A test changes a value of the workflow file by the keyword its variable
spells, lower-cased and without ``QP_``: ``components__reranker__backend``
for ``QP_COMPONENTS__RERANKER__BACKEND``, ``local_models__runtime`` for
``QP_LOCAL_MODELS__RUNTIME``. ``settings_at_paths`` nests such keywords into
the shape the settings take; ``settings_factory`` (conftest) calls it, refuses
a top-level key that is no setting, and the settings merge what it gives into
the workflow file, key by key, so a test names only what it changes.

Example::

    settings_factory(components__reranker__backend="none", run_label="no-rerank")
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from query_processing.config.prompts_file import PROMPTS_FILE_VAR
from query_processing.config.workflow_file import WORKFLOW_FILE_VARIABLE

# Every model component and the vector store on its fake backend: no weights,
# no network, no Elasticsearch.
FAKE_BACKENDS: dict[str, str] = {
    "components__embedder__backend": "fake",
    "components__reranker__backend": "fake",
    "components__llm__backend": "fake",
    "components__vector_store__backend": "fake",
}
# The same, as the variables a process started by a test reads.
FAKE_BACKEND_VARIABLES: dict[str, str] = {
    f"QP_{path.upper()}": value for path, value in FAKE_BACKENDS.items()
}


def shell_settings(
    environ: Mapping[str, str], prefixes: tuple[str, ...] = ("QP_",)
) -> list[str]:
    """The variables of ``environ`` a test clears: each with one of ``prefixes``.

    A variable of the developer's shell would leak into the test. The ones that
    name the tests' workflow and prompts files (conftest.py) stay: the toolkit
    ships none, and a test that clears the others still reads the tests' files.
    """
    kept = {WORKFLOW_FILE_VARIABLE, PROMPTS_FILE_VAR}
    return [name for name in environ if name.startswith(prefixes) and name not in kept]


def without_shell_settings(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` for a process a test starts, without the ``shell_settings``."""
    cleared = set(shell_settings(environ))
    return {name: value for name, value in environ.items() if name not in cleared}


def settings_at_paths(**values: object) -> dict[str, Any]:
    """Nest keywords named by path (levels joined by ``__``) into dictionaries.

    A keyword without ``__`` is a top-level setting, kept as it is; a dictionary
    given for a level is merged with the paths below it.
    """
    nested: dict[str, Any] = {}
    # Dictionaries first, copied, so that a path below one adds to it.
    for path, value in sorted(values.items(), key=lambda item: path_first(item[1])):
        *levels, last = path.split("__")
        node = nested
        for level in levels:
            node = node.setdefault(level, {})
        if isinstance(value, dict):
            node[last] = {**copy.deepcopy(value), **node.get(last, {})}
        else:
            node[last] = value
    return nested


def path_first(value: object) -> bool:
    """Sort key: a dictionary (False) before a plain value (True)."""
    return not isinstance(value, dict)
