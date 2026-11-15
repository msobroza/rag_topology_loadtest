"""The text helpers the ``fake`` backends share: tokens, and cosine.

Example:
    >>> tokenize("Hello, World 42")
    ['hello', 'world', '42']
"""

from __future__ import annotations

import math
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
    """Return the cosine of two vectors: their dot product over both lengths.

    The in-process embedder may return vectors of any length
    (``RAG_EMBEDDING_NORMALIZE`` is off by default), so the length is divided out.

    Args:
        a: A vector.
        b: A vector of the same dimension.

    Returns:
        The cosine of ``a`` and ``b``; 0.0 when either has length 0.

    Raises:
        ValueError: ``a`` and ``b`` differ in dimension.

    Example:
        >>> cosine([3.0, 0.0], [0.5, 0.0])
        1.0
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    lengths = math.hypot(*a) * math.hypot(*b)
    return float(dot / lengths) if lengths else 0.0
