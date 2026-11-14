"""The fake's log format: one JSON object per line, with named fields.

The product's own ``log_json`` is not imported: ``fake_domino`` imports nothing
from ``rag_load_test``.

Example:
    >>> log_event(logging.WARNING, event="app_start_failed", machine="10.10.0.21")
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("fake_domino")


def log_event(level: int = logging.INFO, /, **fields: Any) -> None:
    """Log ``fields`` as one JSON object at ``level`` (INFO unless given first).

    Args:
        level: The ``logging`` level; positional only, so it never clashes with a
            field.
        **fields: The named fields; values that are not JSON are written with
            ``str``.

    Example:
        >>> log_event(event="app_placed", app_id="app-1", machine="10.10.0.21")
    """
    logger.log(level, json.dumps(fields, default=str))
