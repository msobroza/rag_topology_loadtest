"""The Role each App is published as: the workflow app, or a model server.

That domino/app.sh starts each Role is tested in tests/scripts/test_domino_app_sh.py.
"""

from __future__ import annotations

from typing import get_args

from query_processing.domain.roles import MODEL_SERVER_ROLE, WORKFLOW_ROLE, Role


def test_an_app_is_the_workflow_app_or_a_model_server() -> None:
    assert get_args(Role) == ("workflow", "model-server")
    assert get_args(Role) == (WORKFLOW_ROLE, MODEL_SERVER_ROLE)
