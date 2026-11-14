"""Builds the LLM from the settings: ``RAG_GENERATION_BACKEND`` picks which.

Example:
    >>> llm = await LLMFactory(settings, scope, http).open()
"""

from __future__ import annotations

from collections.abc import Mapping

from ..adapters.llm.api.api_llm import open_api_llm
from ..adapters.llm.base_llm import BaseLLM
from ..adapters.llm.fake.fake_llm import FakeLLM
from ..config.rag_settings import RagSettings
from ..infrastructure.http.http_policy import HttpPolicy
from .base_port_factory import BasePortFactory
from .http_resources import HttpResources
from .resource_scope import Opener, ResourceScope


class LLMFactory(BasePortFactory[BaseLLM]):
    """The LLM of each Backend: ``api`` or ``fake``.

    Args:
        settings: The generation Backend, endpoint, timeout and temperature.
        scope: Opens the LLM once.
        http: The shared HTTP client, for the ``api`` Backend.
        supplied: A ready-made LLM, used as given; None to build one.

    Example:
        >>> await LLMFactory(settings, scope, http).open()
    """

    port = "llm"

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        http: HttpResources,
        supplied: BaseLLM | None = None,
    ) -> None:
        super().__init__(settings, scope, supplied)
        self._http = http

    def backend(self) -> str:
        """Return ``RAG_GENERATION_BACKEND``.

        Returns:
            ``api`` or ``fake``.
        """
        return self._settings.generation_backend

    def openers(self) -> Mapping[str, Opener[BaseLLM]]:
        """Return what opens the LLM of each Backend.

        Returns:
            One entry per generation Backend value.
        """
        return {"api": self._api, "fake": self._fake}

    async def _api(self) -> BaseLLM:
        settings = self._settings
        # An answer takes longer than a model-server call: its own timeout, the
        # shared retries and backoff.
        policy = HttpPolicy(
            settings.generation_timeout_s,
            settings.http_retries,
            settings.http_backoff_s,
        )
        return open_api_llm(
            await self._http.client(),
            settings.generation_api(),
            tokens=self._http.run_tokens,
            policy=policy,
            temperature=settings.generation_temperature,
        )

    async def _fake(self) -> BaseLLM:
        return FakeLLM()
