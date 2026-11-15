"""The embedder port: turns a query, or passages, into vectors.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

The two methods are named as LangChain's ``Embeddings`` names them, without its
``a`` for async: every method of a port is async. Some embedding models expect
a prefix in front of a query and none in front of a passage; the prefix belongs
to the model, so an implementation applies it in ``embed_query`` and nowhere
else.

Example:
    >>> class MyEmbedder(BaseEmbedder):
    ...     async def embed_documents(self, texts): ...
    ...     async def embed_query(self, text): ...
    ...     async def ready(self): return True, "mine"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbedder(ABC):
    """Turns a query, or passages, into vectors."""

    model_name: str

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts as they are, with no query prefix: passages to store.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in the same order.
        """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed one query, with the model's query prefix in front if it has one.

        Args:
            text: The query, as the user asked it.

        Returns:
            Its vector.
        """

    @abstractmethod
    async def ready(self) -> tuple[bool, str]:
        """Report whether the embedder can serve requests.

        Returns:
            Whether it is ready, and a short human-readable status such as the
            model name.
        """
