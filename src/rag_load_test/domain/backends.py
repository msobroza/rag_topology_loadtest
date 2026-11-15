"""The vector store's Backend names, the Runtime names, and two defaults.

A Backend is how a component or the vector store runs. A model component's
backends are whatever openers are registered for its type
(``composition/openers/``): ``local``, ``api``, ``fake``, and ``none`` for a
reranker. The vector store's are still a closed set here: ``elasticsearch``,
``chroma`` or ``fake``. Which backend a Topology implies is in
``topology.py``. A Runtime is what runs an in-process model.

Example:
    >>> DEFAULT_VECTOR_STORE_BACKEND
    'elasticsearch'
"""

from __future__ import annotations

from typing import Literal

VectorStoreBackend = Literal["elasticsearch", "chroma", "fake"]
# What runs an in-process model: OpenVINO on its exported files, or PyTorch.
LocalModelRuntime = Literal["openvino", "torch"]

DEFAULT_VECTOR_STORE_BACKEND: VectorStoreBackend = "elasticsearch"
# The run label names an LLM off this backend (run_label.py).
DEFAULT_GENERATION_BACKEND = "api"
