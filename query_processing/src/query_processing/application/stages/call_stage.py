"""The plain call: a stage that only calls its component, bound by its fields.

A plain call is of no stage type. Its binding says which request attributes its
keys fill and which state fields its reads fill, and which state field each
answer attribute is written to; the stage builds the request, awaits the
component's ``ainvoke`` and writes the answer.

Example:
    >>> stage = CallStage(embedder, bind_call(
    ...     BaseEmbedder, {}, StageFields(("standalone_query",), ("query_embedding",))))
    >>> await stage.run({"standalone_query": "vacation days?"})
    {'query_embedding': [...]}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ...domain.base_component import BaseComponent
from ..workflow_state import WorkflowState

if TYPE_CHECKING:  # the binding is built in application, which imports stages
    from ..stage_binding import CallBinding


class CallStage:
    """Call a component with a request built from keys and fields; write its answer.

    Args:
        component: The component the stage calls.
        binding: Where each request attribute comes from, and where each answer
            attribute goes.

    Example:
        >>> CallStage(Upper(), bind_call(
        ...     BaseEcho, {}, StageFields(("message",), ("shout",))))
        <CallStage ...>
    """

    def __init__(
        self, component: BaseComponent[Any, Any], binding: CallBinding
    ) -> None:
        self.component, self.binding = component, binding

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Call the component once.

        Args:
            state: The workflow state; the fields of the binding's reads are read.

        Returns:
            The fields of the binding's writes, each an attribute of the answer.

        Example:
            >>> await CallStage(Upper(), binding).run({"message": "ab"})
            {'shout': 'AB', 'size': 2}
        """
        read = cast(dict[str, Any], state)
        request = self.component.request_type(
            **self.binding.from_keys,
            **{attr: read[field] for attr, field in self.binding.from_reads.items()},
        )
        answer = await self.component.ainvoke(request)
        written = {
            field: getattr(answer, attr)
            for field, attr in self.binding.to_writes.items()
        }
        return cast(WorkflowState, written)
