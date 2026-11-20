"""The import table's exceptions, module by module, on literal sources.

A module the table names alone may import what it names for it and nothing
wider: ``loadtest/target_client.py`` and ``deploy/status.py`` the module of
``composition`` that opens a key source for a command and the key source's
port, ``deploy/workflow_components.py`` the modules of ``composition`` it uses.
The table and its reader are test_layering.py's.
"""

from __future__ import annotations

import pytest
from test_layering import forbidden_imports

TARGET_CLIENT = "query_processing.loadtest.target_client"
STATUS = "query_processing.deploy.status"
WORKFLOW_COMPONENTS = "query_processing.deploy.workflow_components"
OPENS_A_KEY_SOURCE = "from ..composition.standalone_key_source import x"
KEY_SOURCE_PORT = "from ..adapters.key_source.base_key_source import BaseKeySource"
COMPOSITION_ROOT = "from ..composition.composition_root import open_components"


@pytest.mark.parametrize(
    ("module", "source"),
    [
        (TARGET_CLIENT, OPENS_A_KEY_SOURCE),
        (TARGET_CLIENT, KEY_SOURCE_PORT),
        (STATUS, OPENS_A_KEY_SOURCE),
        (STATUS, KEY_SOURCE_PORT),
        (WORKFLOW_COMPONENTS, COMPOSITION_ROOT),
        (WORKFLOW_COMPONENTS, "from ..composition.registered_names import x"),
        (WORKFLOW_COMPONENTS, "from ..config.workflow_settings import x"),
    ],
)
def test_a_module_imports_what_the_table_names_for_it(module: str, source: str) -> None:
    assert forbidden_imports(module, source) == []


@pytest.mark.parametrize(
    ("module", "source"),
    [
        (TARGET_CLIENT, COMPOSITION_ROOT),
        (STATUS, COMPOSITION_ROOT),
        (TARGET_CLIENT, "from ..adapters.llm.base_llm import BaseLlm"),
        (STATUS, "from ..adapters.vector_store.base_vector_store import x"),
        (TARGET_CLIENT, "from ..adapters.key_source.env.env_key_source import x"),
        (TARGET_CLIENT, "from .. import composition"),
        (STATUS, "from ..config.shared_settings import DominoSettings"),
        (WORKFLOW_COMPONENTS, "from ..composition.http_resources import x"),
        (WORKFLOW_COMPONENTS, KEY_SOURCE_PORT),
        ("query_processing.loadtest.qp_record_answers", OPENS_A_KEY_SOURCE),
        ("query_processing.loadtest.qp_record_answers", KEY_SOURCE_PORT),
        ("query_processing.deploy.overrides", OPENS_A_KEY_SOURCE),
        ("query_processing.deploy.publish", KEY_SOURCE_PORT),
    ],
)
def test_an_import_wider_than_the_table_names_is_refused(
    module: str, source: str
) -> None:
    assert forbidden_imports(module, source) != []
