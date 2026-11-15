"""Each component's settings, in the shape the workflow file will give them.

A component's settings are its backend, the settings common to its type, and
one block per backend that has settings, named after that backend: the
embedder's ``local`` block holds what its in-process model takes, its ``api``
block what its endpoint takes. ``WorkflowComponents`` will be the workflow
file's ``components`` key. Until the components move into that file,
``component_settings_of`` derives the shape from the flat ``RAG_*`` variables;
only the ``composition`` package reads it, to open each component.

A backend is a plain name here: the openers registered in ``composition/``
know which ones exist, and refuse any other.

Example:
    >>> components = component_settings_of(RagSettings())
    >>> components.reranker.backend, components.reranker.local.calibration_passages
    ('local', 20)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..domain.api_endpoint import ApiEndpoint
from .rag_settings import RagSettings


class _SettingsBlock(BaseModel):
    # A key the shape does not have is a typo, never a silent default.
    model_config = ConfigDict(frozen=True, extra="forbid")


class ComponentSettings(_SettingsBlock):
    """What every component states: its backend, which picks its opener.

    Example:
        >>> ComponentSettings(backend="fake").backend
        'fake'
    """

    backend: str


class EndpointSettings(_SettingsBlock):
    """An ``api`` block: the endpoint's full route, API format, key and model name.

    Example:
        >>> block = EndpointSettings(url="http://h/v3/rerank", api_format="cohere",
        ...                          api_key="", model="m")
        >>> block.endpoint().api_format
        'cohere'
    """

    url: str
    api_format: str
    api_key: str  # "" = no Authorization header, "domino" = the run's token
    model: str  # the name sent

    def endpoint(self) -> ApiEndpoint:
        """Return the endpoint the block describes.

        Returns:
            Its URL, API format, key and model name.

        Example:
            >>> components.llm.api.endpoint().url
            'https://api.openai.com/v1/chat/completions'
        """
        return ApiEndpoint(self.url, self.api_format, self.api_key, self.model)


class EmbedderEndpointSettings(EndpointSettings):
    """The embedder's ``api`` block: an endpoint, and the texts sent per request."""

    batch_size: int


class LlmEndpointSettings(EndpointSettings):
    """The LLM's ``api`` block: an endpoint, and the temperature sent."""

    temperature: float


class InProcessModelSettings(_SettingsBlock):
    """A ``local`` block: the batch size and threads of an in-process model.

    A value of 0 is calibrated when the component opens.
    """

    batch_size: int
    threads: int


class RerankerLocalSettings(InProcessModelSettings):
    """The reranker's ``local`` block, with the size of its calibration request.

    The request reranks ``calibration_passages`` passages and keeps them all;
    their number also bounds the batch sizes tried.
    """

    calibration_passages: int


class HttpCallSettings(_SettingsBlock):
    """The timeout, retries and backoff of one component's calls."""

    timeout_s: float
    retries: int
    backoff_s: float


class ElasticsearchSettings(_SettingsBlock):
    """The vector store's ``elasticsearch`` block."""

    url: str
    index: str
    api_key: str
    text_analyzer: str


class ChromaSettings(_SettingsBlock):
    """The vector store's ``chroma`` block: a server's URL, else an embedded path."""

    url: str
    path: str
    collection: str


class FakeStoreSettings(_SettingsBlock):
    """The vector store's ``fake`` block: how many synthetic documents seed it."""

    seed_documents: int


class EmbedderSettings(ComponentSettings):
    """The embedder: its weights, its query prefix, a ``local`` and an ``api`` block.

    Example:
        >>> component_settings_of(RagSettings()).embedder.query_prefix
        'query: '
    """

    source_id: str  # the Hugging Face id (or folder) of the in-process weights
    query_prefix: str  # put in front of a query, never of a passage
    local: InProcessModelSettings
    api: EmbedderEndpointSettings


class RerankerSettings(ComponentSettings):
    """The reranker: its weights, a ``local`` and an ``api`` block."""

    source_id: str
    local: RerankerLocalSettings
    api: EndpointSettings


class LlmSettings(ComponentSettings):
    """The LLM: an ``api`` block, and the HTTP settings of its calls."""

    api: LlmEndpointSettings
    http: HttpCallSettings


class VectorStoreSettings(ComponentSettings):
    """The vector store: one block per product, and the fake's."""

    elasticsearch: ElasticsearchSettings
    chroma: ChromaSettings
    fake: FakeStoreSettings


class WorkflowComponents(_SettingsBlock):
    """The workflow file's ``components``: one entry per component, by name."""

    embedder: EmbedderSettings
    reranker: RerankerSettings
    llm: LlmSettings
    vector_store: VectorStoreSettings


def component_settings_of(settings: RagSettings) -> WorkflowComponents:
    """Derive every component's settings from the flat ``RAG_*`` settings.

    Nothing is validated again: ``settings`` was, when it was built, except a
    copy made with ``model_copy``, which keeps whatever names it was given.

    Args:
        settings: The flat settings.

    Returns:
        The four components, in the workflow file's shape.

    Example:
        >>> component_settings_of(RagSettings()).llm.http.timeout_s
        120.0
    """
    return WorkflowComponents(
        embedder=_embedder_of(settings),
        reranker=_reranker_of(settings),
        llm=_llm_of(settings),
        vector_store=_vector_store_of(settings),
    )


def _embedder_of(settings: RagSettings) -> EmbedderSettings:
    endpoint = settings.embedding_api()._asdict()
    return EmbedderSettings(
        backend=settings.embedding_backend,
        source_id=settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        local=InProcessModelSettings(
            batch_size=settings.embedding_batch_size,
            threads=settings.embedding_threads,
        ),
        api=EmbedderEndpointSettings(
            **endpoint, batch_size=settings.embedding_batch_size
        ),
    )


def _reranker_of(settings: RagSettings) -> RerankerSettings:
    # The calibration request is sized by the reranker's own setting: the code
    # that opens a component cannot read the settings of the stage calling it.
    local = RerankerLocalSettings(
        batch_size=settings.rerank_batch_size,
        threads=settings.rerank_threads,
        calibration_passages=settings.rerank_calibration_passages,
    )
    return RerankerSettings(
        backend=settings.rerank_backend,
        source_id=settings.rerank_model,
        local=local,
        api=EndpointSettings(**settings.rerank_api()._asdict()),
    )


def _llm_of(settings: RagSettings) -> LlmSettings:
    # An answer takes longer than a model-server call: its own timeout, the
    # shared retries and backoff.
    http = HttpCallSettings(
        timeout_s=settings.generation_timeout_s,
        retries=settings.http_retries,
        backoff_s=settings.http_backoff_s,
    )
    endpoint = settings.generation_api()._asdict()
    return LlmSettings(
        backend=settings.generation_backend,
        api=LlmEndpointSettings(
            **endpoint, temperature=settings.generation_temperature
        ),
        http=http,
    )


def _vector_store_of(settings: RagSettings) -> VectorStoreSettings:
    return VectorStoreSettings(
        backend=settings.vector_store_backend,
        elasticsearch=ElasticsearchSettings(
            url=settings.es_url,
            index=settings.es_index,
            api_key=settings.es_api_key,
            text_analyzer=settings.es_text_analyzer,
        ),
        chroma=ChromaSettings(
            url=settings.chroma_url,
            path=settings.chroma_path,
            collection=settings.chroma_collection,
        ),
        fake=FakeStoreSettings(seed_documents=settings.fake_store_seed_documents),
    )
