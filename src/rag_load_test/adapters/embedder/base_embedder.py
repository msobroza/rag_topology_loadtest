"""The embedder port: turns texts into vectors.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

Example:
    >>> class MyEmbedder(BaseEmbedder):
    ...     async def embed(self, texts): ...
    ...     async def ready(self): return True, "mine"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbedder(ABC):
    """Turns texts into vectors."""

    model_name: str

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in the same order.
        """

    @abstractmethod
    async def ready(self) -> tuple[bool, str]:
        """Report whether the embedder can serve requests.

        Returns:
            Whether it is ready, and a short human-readable status such as the
            model name.
        """
