"""The Runtime names: what runs an in-process model.

A Backend is how a component runs. The backends of each component type are
whatever openers are registered for it (``composition/openers/``): for a
model, ``local``, ``served`` (a model server of ours), ``api`` (an endpoint
we do not run), ``fake``, and ``none`` for a reranker; for the vector store,
``elasticsearch``, ``chroma`` and ``fake``. Each component's own settings give
its backend, and nothing else decides it: the deploy tool writes the backends a
topology alias gives (ADR 0003). A Runtime is what runs an in-process model.

Example:
    >>> from typing import get_args
    >>> get_args(LocalModelRuntime)
    ('openvino', 'torch')
"""

from __future__ import annotations

from typing import Literal

# What runs an in-process model: OpenVINO on its exported files, or PyTorch.
LocalModelRuntime = Literal["openvino", "torch"]
