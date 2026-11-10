"""What one workflow needs besides its ports: retrieval sizes and prompts.

``RagSettings.workflow_parameters()`` builds it from the configuration.

Example:
    >>> RagWorkflow(deps, settings.workflow_parameters())
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import RagPrompts


@dataclass(frozen=True)
class WorkflowParameters:
    """Retrieval sizes and prompts of one workflow.

    ``top_k_retrieve`` candidates come from the store, ``top_k_rerank`` are kept for
    the LLM, and ``prompts`` frame them.
    """

    top_k_retrieve: int
    top_k_rerank: int
    prompts: RagPrompts
