"""What one workflow needs besides its ports: retrieval sizes and prompts.

``RagSettings.workflow_parameters()`` builds it from the configuration.

Example::

    RagWorkflow(deps, settings.workflow_parameters())
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import RagPrompts


@dataclass(frozen=True)
class WorkflowParameters:
    """``top_k_retrieve`` candidates from the store, ``top_k_rerank`` kept for the
    LLM, and the prompts that frame them."""

    top_k_retrieve: int
    top_k_rerank: int
    prompts: RagPrompts
