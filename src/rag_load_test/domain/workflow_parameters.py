"""What one workflow needs besides its ports: how it retrieves, and its prompts.

``RagSettings.workflow_parameters()`` builds it from the configuration.

Example:
    >>> RagWorkflow(deps, settings.workflow_parameters())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .prompts import RagPrompts

# How the retrieve stage finds candidates: by vector, by words, or both
# fused by reciprocal rank.
RetrievalMode = Literal["dense", "lexical", "hybrid"]
DEFAULT_RETRIEVAL_MODE: RetrievalMode = "dense"


@dataclass(frozen=True)
class WorkflowParameters:
    """Retrieval mode, sizes and prompts of one workflow.

    ``top_k_retrieve`` candidates come from the store by ``retrieval_mode``,
    ``top_k_rerank`` are kept for the LLM, and ``prompts`` frame them.
    ``rrf_k`` is the rank constant of the fusion a ``hybrid`` retrieval ends
    with. The query prefix is not here: it belongs to the embedding model, and
    the embedder applies it.
    """

    top_k_retrieve: int
    top_k_rerank: int
    prompts: RagPrompts
    retrieval_mode: RetrievalMode
    rrf_k: int
