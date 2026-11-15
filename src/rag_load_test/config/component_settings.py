"""Each component's settings: the workflow file's ``components`` block.

A component is named. Its type is the port it is used through (``embedder``,
``reranker``, ``llm``) and defaults to its name, so two components of one type
take two names and state ``type``. It states its backend, which picks its
opener; the openers registered in ``composition/`` know which backends exist,
and refuse any other. A model component has a source id (the weights) and an
alias (the name a model server serves it under), which defaults to the last
part of the source id. The settings common to its type sit beside them, and
those of one backend in a block named after it: the embedder's ``local``
block holds what its in-process model takes, its ``api`` block what its
endpoint takes. A key the type does not have, a block of a backend it does
not have included, is refused.

The vector store is still read from flat ``RAG_*`` variables; its settings
take the same shape here, and ``RagSettings`` builds them.

Example:
    >>> EmbedderSettings(backend="local", source_id="Org/model-x").alias
    'model-x'
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.api_endpoint import ApiEndpoint

# The name of the vector store, which no file declares yet: it is the
# component a retrieve stage calls, read from the flat RAG_* variables.
VECTOR_STORE = "vector_store"


class _SettingsBlock(BaseModel):
    # A key the shape does not have is a typo, never a silent default.
    model_config = ConfigDict(frozen=True, extra="forbid")


class EndpointSettings(_SettingsBlock):
    """An ``api`` block: the endpoint's full route, API format, key and model name.

    Example:
        >>> EndpointSettings(url="http://h/v3/rerank", api_format="cohere"
        ...                  ).endpoint("alias").model
        'alias'
    """

    url: str = ""  # the full route, e.g. http://host:8001/v3/rerank
    # A format registered in adapters/<component>/api/. The settings cannot
    # import that registry, so each command refuses an unknown name before it
    # opens anything (composition/registered_names.py).
    api_format: str
    # "" = no Authorization header, "domino" = the run's token, else a bearer
    # token. Read from the environment: a key is never written in the file.
    api_key: str = ""
    model: str = ""  # the name sent; "" = the component's alias

    def endpoint(self, alias: str) -> ApiEndpoint:
        """Return the endpoint the block describes.

        Args:
            alias: The component's alias, sent when the block names no model:
                our model servers serve a model under its alias.

        Returns:
            Its URL, API format, key and model name.

        Example:
            >>> components["llm"].api.endpoint("").url
            'https://api.openai.com/v1/chat/completions'
        """
        return ApiEndpoint(self.url, self.api_format, self.api_key, self.model or alias)


class EmbedderEndpointSettings(EndpointSettings):
    """The embedder's ``api`` block: an endpoint, and the texts sent per request."""

    api_format: str = "openai"
    batch_size: int = Field(default=64, ge=1)


class RerankerEndpointSettings(EndpointSettings):
    """The reranker's ``api`` block: an endpoint in a rerank API format."""

    api_format: str = "cohere"


class LlmEndpointSettings(EndpointSettings):
    """The LLM's ``api`` block: an endpoint, and the temperature sent."""

    api_format: str = "openai"
    temperature: float = Field(default=0.0, ge=0, le=2)


class EmbedderLocalSettings(_SettingsBlock):
    """The embedder's ``local`` block: its in-process model's batch size and threads.

    A query is one text, so the batch size is never calibrated; threads at 0
    are, when the component opens.
    """

    batch_size: int = Field(default=64, ge=1)  # texts per encode call
    threads: int = Field(default=0, ge=0)  # 0 = calibrated at start-up


class RerankerLocalSettings(_SettingsBlock):
    """The reranker's ``local`` block, with the size of its calibration request.

    A batch size or thread count of 0 is calibrated when the component opens.
    The calibration request reranks ``calibration_passages`` passages and keeps
    them all; their number also bounds the batch sizes tried.
    """

    batch_size: int = Field(default=0, ge=0)  # passages per predict batch
    threads: int = Field(default=0, ge=0)
    # The reranker's own: calibration reads no stage's setting.
    calibration_passages: int = Field(default=20, ge=1)


class LlmHttpSettings(_SettingsBlock):
    """The timeout of the LLM's calls: an answer takes longer than a model-server call.

    Its retries and backoff are the shared ``RAG_HTTP_RETRIES`` and
    ``RAG_HTTP_BACKOFF_S``.
    """

    timeout_s: float = Field(default=120.0, gt=0)


class ComponentSettings(_SettingsBlock):
    """What every component states: its backend, which picks its opener.

    Example:
        >>> EmbedderSettings(backend="fake", source_id="org/m").backend
        'fake'
    """

    type: str
    backend: str

    @model_validator(mode="before")
    @classmethod
    def refuse_unknown_keys(cls, data: Any) -> Any:
        """Refuse a key the type does not have, naming the keys it has.

        Args:
            data: The component's entry, as read.

        Returns:
            ``data``, unchanged.

        Raises:
            ValueError: If a key is not a setting or block of the type, such as
                the ``local`` block of an LLM.
        """
        if not isinstance(data, Mapping):
            return data
        unknown = sorted(str(key) for key in data if key not in cls.model_fields)
        if unknown:
            raise ValueError(
                f"the {cls.model_fields['type'].default!r} component type has no "
                f"{', '.join(map(repr, unknown))}; expected its keys "
                f"({', '.join(sorted(cls.model_fields))}), a backend's settings "
                "in the block named after that backend"
            )
        return data


class ModelComponentSettings(ComponentSettings):
    """A model: its source id, and the alias a model server serves it under."""

    source_id: str = ""  # a Hugging Face id, or a folder holding the weights
    alias: str = ""  # "" = the last part of the source id

    @model_validator(mode="before")
    @classmethod
    def alias_defaults_to_source_id(cls, data: Any) -> Any:
        """Default the alias to the last part of the source id.

        Args:
            data: The component's entry, as read.

        Returns:
            ``data``, with ``alias`` set when it was missing or empty.

        Example:
            >>> RerankerSettings(backend="local", source_id="/models/rr/").alias
            'rr'
        """
        if not isinstance(data, Mapping) or data.get("alias"):
            return data
        return {**data, "alias": alias_of(str(data.get("source_id") or ""))}


class EmbedderSettings(ModelComponentSettings):
    """An embedder: its weights, its query prefix, a ``local`` and an ``api`` block.

    Example:
        >>> settings.components["embedder"].query_prefix
        'query: '
    """

    type: Literal["embedder"] = "embedder"
    source_id: str
    # Put in front of every query before it is embedded, never of a passage
    # (arctic-embed expects "query: "); "" for a model that takes none.
    query_prefix: str = ""
    local: EmbedderLocalSettings = Field(default_factory=EmbedderLocalSettings)
    api: EmbedderEndpointSettings = Field(default_factory=EmbedderEndpointSettings)


class RerankerSettings(ModelComponentSettings):
    """A reranker: its weights, a ``local`` and an ``api`` block."""

    type: Literal["reranker"] = "reranker"
    source_id: str
    local: RerankerLocalSettings = Field(default_factory=RerankerLocalSettings)
    api: RerankerEndpointSettings = Field(default_factory=RerankerEndpointSettings)


class LlmSettings(ModelComponentSettings):
    """An LLM: an ``api`` block, and the timeout of its calls. It has no ``local``."""

    type: Literal["llm"] = "llm"
    api: LlmEndpointSettings = Field(default_factory=LlmEndpointSettings)
    http: LlmHttpSettings = Field(default_factory=LlmHttpSettings)


# A model component of the workflow file, its settings chosen by its type.
ModelComponent = Annotated[
    EmbedderSettings | RerankerSettings | LlmSettings, Field(discriminator="type")
]


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


class VectorStoreSettings(ComponentSettings):
    """The vector store: one block per product, and the fake's."""

    type: Literal["vector_store"] = "vector_store"
    elasticsearch: ElasticsearchSettings
    chroma: ChromaSettings
    fake: FakeStoreSettings


def alias_of(source_id: str) -> str:
    """Return the alias a source id gives by default: its last part.

    Args:
        source_id: A Hugging Face id or a folder.

    Returns:
        The name after the last ``/``, a trailing ``/`` ignored; "" for "".

    Example:
        >>> alias_of("Snowflake/snowflake-arctic-embed-l-v2.0")
        'snowflake-arctic-embed-l-v2.0'
    """
    return PurePosixPath(source_id).name


def typed_by_name(entries: object) -> object:
    """Give each component entry without a ``type`` its name as type.

    Args:
        entries: The ``components`` block, as read.

    Returns:
        The block, each entry that is a mapping and names no type given its
        component's name as type; anything else unchanged, for pydantic to
        refuse.

    Example:
        >>> typed_by_name({"llm": {"backend": "fake"}})
        {'llm': {'type': 'llm', 'backend': 'fake'}}
    """
    if not isinstance(entries, Mapping):
        return entries
    return {name: _typed(name, entry) for name, entry in entries.items()}


def _typed(name: object, entry: object) -> object:
    if isinstance(entry, Mapping) and "type" not in entry:
        return {"type": name, **entry}
    return entry
