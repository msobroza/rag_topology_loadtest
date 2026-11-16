"""The tests the toolkit ships, which a user runs on a component of their own.

A package of the product, not of ``tests/``, so a use case imports it, through
``extension_api``. It may import every package of the toolkit, and none
imports it.

Example:
    >>> from query_processing.testing import ComponentContract
"""

from __future__ import annotations

from .component_contract import ComponentContract

__all__ = ["ComponentContract"]
