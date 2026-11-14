"""The app's one log format: a JSON object per line, with named fields.

Example:
    >>> log_json(event="rag_startup", topology="monolith")
    >>> log_json(logging.WARNING, event="hardware_tier_cpus_unavailable", status=403)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("rag_load_test")


def log_json(level: int = logging.INFO, /, **fields: Any) -> None:
    """Log ``fields`` as one JSON object at ``level`` (INFO unless given first).

    Args:
        level: The ``logging`` level; positional only, so it never clashes with a
            field.
        **fields: The named fields of the JSON object; values that are not JSON
            are written with ``str``.

    Example:
        >>> log_json(event="thread_budget", cpus=4.0, workers=4)
    """
    logger.log(level, json.dumps(fields, default=str))
