"""The Role each App is published as, and the App parts of a topology.

An App is the workflow app or a model server. ``domino/app.sh`` reads the Role
from ``QP_ROLE``: the workflow app runs ``qp-serve``; a model server is told
the component it serves and its model-server kind, whose own launch script
starts it (``domino/model_servers/<kind>.sh``). Which Apps a topology needs is
the deploy tool's to say: one model server App per component its topology alias
serves, then the workflow App (``deploy/topology_alias.py``).

Example:
    >>> WORKFLOW_ROLE, MODEL_SERVER_ROLE
    ('workflow', 'model-server')
"""

from __future__ import annotations

from typing import Literal

# What one Domino App is published as: QP_ROLE, read by domino/app.sh.
Role = Literal["workflow", "model-server"]
WORKFLOW_ROLE: Role = "workflow"
MODEL_SERVER_ROLE: Role = "model-server"
# Which App of a topology it is: the workflow App, or the model server of one
# served component, named after that component.
AppPart = str
WORKFLOW_APP = "workflow"
