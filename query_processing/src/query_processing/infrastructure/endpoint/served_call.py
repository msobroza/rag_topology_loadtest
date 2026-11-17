"""``ServedCallMixin``: a component a model server of ours answers, on its call route.

The toolkit builds the ``served`` class of a registered component type from it
and the type's abstract class (``composition/registered_components.py``). A
use case that redefines that class subclasses it too, so the composition
builds both alike: with the endpoint client of the component's ``served`` URL
and the component type's API format.

Example:
    >>> served = ServedShouter(endpoint_client, api_format_of(BaseShouter))
    >>> (await served.ainvoke(ShoutRequest(text="a"))).text
    'A!'
"""

from __future__ import annotations

from typing import Any

from ...domain.api_format import BaseApiFormat
from .api_endpoint_client import ApiEndpointClient


class ServedCallMixin:
    """Answers ``ainvoke`` by posting the request to a model server's call route.

    It comes before the component type's abstract class in the bases, so its
    ``ainvoke`` is the one called.

    Args:
        endpoint_client: The call route, how patiently it is called, and the
            component's name, which every error names.
        api_format: Puts the request on the wire and reads the answer.

    Example:
        >>> class ServedShouter(ServedCallMixin, BaseShouter): ...
    """

    def __init__(
        self, endpoint_client: ApiEndpointClient, api_format: BaseApiFormat[Any, Any]
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format

    async def ainvoke(self, request: Any) -> Any:
        """Post the request and read the answer.

        Args:
            request: The component type's request.

        Returns:
            The component type's answer.

        Raises:
            RagDependencyError: If the call fails, or the answer is not the
                API format's shape; it names the component.

        Example:
            >>> await served.ainvoke(ShoutRequest(text="a"))
            Shouted(text='A!')
        """
        wire = self._api_format.request_to_wire(request)
        body = await self._endpoint_client.post(wire)
        try:
            return self._api_format.answer_from_wire(body)
        except ValueError as exc:
            raise self._endpoint_client.malformed(exc) from exc
