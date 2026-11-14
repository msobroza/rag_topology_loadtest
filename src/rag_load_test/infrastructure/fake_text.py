"""The text helpers the ``fake`` backends share: tokens, and cosine.

Example:
    >>> tokenize("Hello, World 42")
    ['hello', 'world', '42']
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lower-case alphanumeric tokens.

    Args:
        text: The text to split.

    Returns:
        The runs of ``[a-z0-9]`` in the lower-cased text, in order.

    Example:
        >>> tokenize("Hello, World 42")
        ['hello', 'world', '42']
    """
    return _TOKEN.findall(text.lower())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product, which is the cosine for unit vectors.

    Args:
        a: A unit vector.
        b: A unit vector of the same length.

    Returns:
        The dot product of ``a`` and ``b``.
    """
    return float(sum(x * y for x, y in zip(a, b, strict=True)))
