"""The retrieval mode: how the retrieve stage finds its candidates.

``dense`` asks the store by vector, ``lexical`` by words, ranked by the store,
and ``hybrid`` does both and fuses the two rankings by reciprocal rank. It is a
setting of the retrieve stage.

Example:
    >>> DEFAULT_RETRIEVAL_MODE
    'dense'
"""

from __future__ import annotations

from typing import Literal

RetrievalMode = Literal["dense", "lexical", "hybrid"]
DEFAULT_RETRIEVAL_MODE: RetrievalMode = "dense"
