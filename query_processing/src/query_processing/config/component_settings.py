"""Each component's settings: the workflow file's ``components`` block.

A component is named. Its type is the port it is used through (``embedder``,
``reranker``, ``llm``, ``vector_store``) and defaults to its name, so two
components of one type take two names and state ``type``. It is read with
the settings class its type names in ``COMPONENT_TYPE_SETTINGS``, else, for a
component type registered in ``COMPONENT_TYPES`` (one an extension defines),
with ``RegisteredComponentSettings``; any other type is refused, listing the
known types. It states its
backend, which picks its opener; the openers registered in ``composition/``
know which backends exist, and refuse any other. A model component has a
source id (the weights) and an alias (the name a model server serves it
under), which defaults to the last part of the source id. The settings common
to its type sit beside them, and those of one backend in a block named after
it: the embedder's ``local`` block holds what its in-process model takes, its
``served`` block the route of the model server of ours that serves it, its
``api`` block what an endpoint we do not run takes. How patiently a component
on ``served`` or ``api`` is called is its ``http`` block, whichever of the two
it is on: its timeout, retries and backoff, each left out taking the default
shared by every component (``shared_settings.py``). The vector store's
backends are named after their product, ``elasticsearch`` and ``chroma``, and
``fake``, each with its block. A key the type does not have, a block of a
backend it does not have included, is refused.

Example:
    >>> EmbedderSettings(backend="local", source_id="Org/model-x").alias
    'model-x'
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, PlainValidator, SerializeAsAny, model_validator

from ..domain.api_endpoint import ApiEndpoint
from ..domain.base_component import COMPONENT_TYPES
from ..domain.registry import Registry
from ..domain.standard_formats import STANDARD_API_FORMATS
from .registered_component_settings import RegisteredComponentSettings
from .settings_blocks import (
    ComponentHttpSettings,
    ComponentSettings,
    ServedSettings,
    SettingsBlock,
)


class EndpointSettings(ServedSettings):
    """An ``api`` block: an endpoint we do not run, its API format, key and model name.

    It holds what a ``served`` block holds, plus what a served component takes
    from its type, its alias and where the app runs: the format, model and key.

    Example:
        >>> EndpointSettings(url="http://h/v3/rerank", api_format="cohere"
        ...                  ).endpoint("alias").model
        'alias'
    """

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
            alias: The component's alias, sent when the block names no model.

        Returns:
            Its URL, API format, key and model name.

        Example:
            >>> components["llm"].api.endpoint("").url
            'https://api.openai.com/v1/chat/completions'
        """
        return ApiEndpoint(self.url, self.api_format, self.api_key, self.model or alias)


class EmbedderServedSettings(ServedSettings):
    """The embedder's ``served`` block: a route, and the texts sent per request."""

    batch_size: int = Field(default=64, ge=1)


class EmbedderEndpointSettings(EndpointSettings, EmbedderServedSettings):
    """The embedder's ``api`` block: an endpoint, and the texts sent per request."""

    api_format: str = STANDARD_API_FORMATS["embedder"]


class RerankerEndpointSettings(EndpointSettings):
    """The reranker's ``api`` block: an endpoint in a rerank API format."""

    api_format: str = STANDARD_API_FORMATS["reranker"]


class LlmServedSettings(ServedSettings):
    """The LLM's ``served`` block: a route, and the temperature sent."""

    temperature: float = Field(default=0.0, ge=0, le=2)


class LlmEndpointSettings(EndpointSettings, LlmServedSettings):
    """The LLM's ``api`` block: an endpoint, and the temperature sent."""

    api_format: str = STANDARD_API_FORMATS["llm"]


class EmbedderLocalSettings(SettingsBlock):
    """The embedder's ``local`` block: its in-process model's batch size and threads.

    A query is one text, so the batch size is never calibrated; threads at 0
    are, when the component opens. ``normalize`` says whether its vectors are
    scaled to unit length.
    """

    batch_size: int = Field(default=64, ge=1)  # texts per encode call
    threads: int = Field(default=0, ge=0)  # 0 = calibrated at start-up
    # Scale each vector the in-process embedder returns to unit length. Off is
    # safe: every store compares vectors by cosine, which ignores their length.
    normalize: bool = False


class RerankerLocalSettings(SettingsBlock):
    """The reranker's ``local`` block, with the size of its calibration request.

    A batch size or thread count of 0 is calibrated when the component opens.
    The calibration request reranks ``calibration_passages`` passages and keeps
    them all; their number also bounds the batch sizes tried.
    """

    batch_size: int = Field(default=0, ge=0)  # passages per predict batch
    threads: int = Field(default=0, ge=0)
    # The reranker's own: calibration reads no stage's setting.
    calibration_passages: int = Field(default=20, ge=1)


class ModelComponentSettings(ComponentSettings):
    """A model: its source id, its alias, and how patiently it is called over HTTP."""

    source_id: str = ""  # a Hugging Face id, or a folder holding the weights
    alias: str = ""  # "" = the last part of the source id
    http: ComponentHttpSettings = Field(default_factory=ComponentHttpSettings)

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
    """An embedder: its weights, its query prefix, and one block per backend.

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
    served: EmbedderServedSettings = Field(default_factory=EmbedderServedSettings)
    api: EmbedderEndpointSettings = Field(default_factory=EmbedderEndpointSettings)


class RerankerSettings(ModelComponentSettings):
    """A reranker: its weights, a ``local``, a ``served`` and an ``api`` block."""

    type: Literal["reranker"] = "reranker"
    source_id: str
    local: RerankerLocalSettings = Field(default_factory=RerankerLocalSettings)
    served: ServedSettings = Field(default_factory=ServedSettings)
    api: RerankerEndpointSettings = Field(default_factory=RerankerEndpointSettings)


class LlmSettings(ModelComponentSettings):
    """An LLM: a ``served`` and an ``api`` block.

    It has no ``local`` block: no LLM runs in the app's process.
    """

    type: Literal["llm"] = "llm"
    served: LlmServedSettings = Field(default_factory=LlmServedSettings)
    api: LlmEndpointSettings = Field(default_factory=LlmEndpointSettings)


# A model component: what a model server can serve or an endpoint answer for.
ModelComponent = EmbedderSettings | RerankerSettings | LlmSettings


class ElasticsearchSettings(SettingsBlock):
    """The vector store's ``elasticsearch`` block: kNN, and BM25 for lexical search.

    Example:
        >>> ElasticsearchSettings(index="passages_v2").url
        'http://localhost:9200'
    """

    url: str = "http://localhost:9200"  # user:password@ in the URL for basic auth
    # The use case's own: "" = unset, refused on this backend (WorkflowSettings).
    index: str = ""
    # "" = none (basic auth in the URL, or no security). Read from the
    # environment: a key is never written in the file.
    api_key: str = ""
    # Analyzer of the passages' text, applied when the index is created and to
    # every lexical query: "standard", or a language's (e.g. "french"). An
    # index keeps the analyzer it was created with.
    text_analyzer: str = "standard"


class ChromaSettings(SettingsBlock):
    """The vector store's ``chroma`` block: a server's URL, else an embedded path.

    Example:
        >>> ChromaSettings().path
        'chroma'
    """

    url: str = ""  # a Chroma server; "" = embedded Chroma at path
    path: str = "chroma"  # in the working directory, the use case's
    # The use case's own: "" = unset, refused on this backend (WorkflowSettings).
    collection: str = ""


class FakeStoreSettings(SettingsBlock):
    """The vector store's ``fake`` block: how many synthetic documents seed it.

    Example:
        >>> FakeStoreSettings().seed_documents
        200
    """

    # Same size as ``make ingest`` so a fake-store app answers like an ingested one.
    seed_documents: int = Field(default=200, ge=1)


class VectorStoreSettings(ComponentSettings):
    """The vector store: one block per product, and the fake's.

    The index or collection the passages are kept under is the use case's to
    name: the block of the backend in use states it, or start-up refuses it.

    Example:
        >>> settings.components["vector_store"].elasticsearch.index
        'rag_passages'
    """

    type: Literal["vector_store"] = "vector_store"
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    # "fake" = in memory, seeded with the synthetic corpus when it opens
    fake: FakeStoreSettings = Field(default_factory=FakeStoreSettings)


# The settings class of each component type, found by the type's name: a
# component is read with the class its type names here. The toolkit registers
# the class of a type of its own here, since a port cannot import the config.
COMPONENT_TYPE_SETTINGS: Registry[type[ComponentSettings]] = Registry("component type")
# Registered by call, not as a decorator: the decorator returns the registry's
# entry type, which would hide each class's own fields from the type checkers.
COMPONENT_TYPE_SETTINGS.register("embedder")(EmbedderSettings)
COMPONENT_TYPE_SETTINGS.register("reranker")(RerankerSettings)
COMPONENT_TYPE_SETTINGS.register("llm")(LlmSettings)
COMPONENT_TYPE_SETTINGS.register("vector_store")(VectorStoreSettings)


def read_component(entry: object) -> ComponentSettings:
    """Read a component's entry with the settings class its type names.

    Args:
        entry: The entry as read, its ``type`` already given (``typed_by_name``),
            or settings already read.

    Returns:
        The component's settings, of the class registered for its type, else a
        ``RegisteredComponentSettings`` for a type registered in
        ``COMPONENT_TYPES``.

    Raises:
        ValueError: If the entry is no mapping, or its type is unknown; the
            message lists the known types.

    Example:
        >>> read_component({"type": "llm", "backend": "fake"}).backend
        'fake'
    """
    if isinstance(entry, ComponentSettings):
        return entry
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"a component's entry is {entry!r}; expected a mapping of its keys, "
            "such as {backend: fake}"
        )
    type_name = str(entry.get("type"))
    if type_name in COMPONENT_TYPE_SETTINGS.names():
        return COMPONENT_TYPE_SETTINGS.get(type_name).model_validate(entry)
    if type_name in COMPONENT_TYPES.names():
        return RegisteredComponentSettings.model_validate(entry)
    known = sorted({*COMPONENT_TYPE_SETTINGS.names(), *COMPONENT_TYPES.names()})
    raise ValueError(
        f"unknown component type {type_name!r} (expected one of: "
        f"{', '.join(map(repr, known))})"
    )


# A component of the workflow file, read with the settings class of its type.
# Dumped with the keys of that class, not only those of ComponentSettings.
Component = Annotated[SerializeAsAny[ComponentSettings], PlainValidator(read_component)]


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
