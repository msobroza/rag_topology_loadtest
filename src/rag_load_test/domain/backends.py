"""The Runtime names, and the default Backends the run label is measured against.

A Backend is how a component runs. The backends of each component type are
whatever openers are registered for it (``composition/openers/``): for a
model, ``local``, ``served`` (a model server of ours), ``api`` (an endpoint
we do not run), ``fake``, and ``none`` for a reranker; for the vector store,
``elasticsearch``, ``chroma`` and ``fake``. Which backend a Topology implies
is in ``topology.py``. A Runtime is what runs an in-process model.

Example:
    >>> DEFAULT_VECTOR_STORE_BACKEND
    'elasticsearch'
"""

from __future__ import annotations

from typing import Literal

# What runs an in-process model: OpenVINO on its exported files, or PyTorch.
LocalModelRuntime = Literal["openvino", "torch"]

# The run label names a vector store and an LLM off these backends (run_label.py).
DEFAULT_VECTOR_STORE_BACKEND = "elasticsearch"
DEFAULT_GENERATION_BACKEND = "api"
