"""The tests the toolkit ships, which a user runs on a component of their own.

A package of the product, not of ``tests/``, so a use case's tests import it
from here: ``ComponentContract`` for any component type, ``RouterContract``
for a router, ``check_pinned_answers`` for a pinned answer record,
``registered_orchestrators`` for the orchestrators a test runs on. It may
import every package of the toolkit, and none imports it, ``extension_api``
included, so that no command loads the model server and
the test client it serves a component with.

Example:
    >>> from query_processing.testing import ComponentContract, RouterContract
"""

from __future__ import annotations

from .component_contract import ComponentContract
from .directory_size import (
    FolderNotYetSplit,
    OnePatternFolder,
    directory_size_findings,
)
from .pinned_answer_record import (
    PinnedAnswerRecord,
    PinnedAnswersDifferError,
    add_pin_answers_option,
    check_pinned_answers,
    clear_loadtest_variables,
    compare_answer_records,
    pin_answers_asked,
    record_answers_of,
    use_loadtest_variables,
    workflow_client,
)
from .registered_orchestrators import registered_orchestrators
from .router_contract import RouterContract

__all__ = [
    "ComponentContract",
    "FolderNotYetSplit",
    "OnePatternFolder",
    "PinnedAnswerRecord",
    "PinnedAnswersDifferError",
    "RouterContract",
    "add_pin_answers_option",
    "check_pinned_answers",
    "clear_loadtest_variables",
    "compare_answer_records",
    "directory_size_findings",
    "pin_answers_asked",
    "record_answers_of",
    "registered_orchestrators",
    "use_loadtest_variables",
    "workflow_client",
]
