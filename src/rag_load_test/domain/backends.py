"""The Backend and Runtime names: how a component or the vector store runs.

A Backend is how a component (embedding, rerank, generation) or the vector store
runs: ``local`` (a model loaded in the workflow app's process), ``api`` (an
endpoint called over HTTP), ``fake``, ``none`` (rerank only), and for the store
``elasticsearch``, ``chroma`` or ``fake``. The variants load no model of ours into
the process, so every Topology allows them; which backend a Topology implies is
in ``topology.py``. A Runtime is what runs an in-process model; a Weight format
is what an OpenVINO export stores its weights as.

Example:
    >>> "none" in RERANK_VARIANTS
    True
"""

from __future__ import annotations

from typing import Literal

EmbeddingBackend = Literal["local", "api", "fake"]
RerankBackend = Literal["local", "api", "fake", "none"]
GenerationBackend = Literal["api", "fake"]
VectorStoreBackend = Literal["elasticsearch", "chroma", "fake"]
# The stores that rank text themselves; open-source Chroma does not (Cloud only).
LEXICAL_STORE_BACKENDS: tuple[VectorStoreBackend, ...] = ("elasticsearch", "fake")
# What runs an in-process model: OpenVINO on its exported files, or PyTorch.
LocalModelRuntime = Literal["openvino", "torch"]
# What an OpenVINO export stores its weights as: 32-bit floats, or 8-bit integers
# (weight compression only, not full quantization). Both exports read the one
# setting, the in-process one and OVMS's, so servers compare on equal weights.
OpenVinoWeightFormat = Literal["fp32", "int8"]

# Allowed in every topology: they load no model of ours into this process.
EMBEDDING_VARIANTS: tuple[EmbeddingBackend, ...] = ("api", "fake")
RERANK_VARIANTS: tuple[RerankBackend, ...] = ("api", "fake", "none")

DEFAULT_VECTOR_STORE_BACKEND: VectorStoreBackend = "elasticsearch"
DEFAULT_GENERATION_BACKEND: GenerationBackend = "api"
