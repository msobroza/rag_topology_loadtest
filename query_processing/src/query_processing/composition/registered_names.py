"""An unknown registered name, refused before a command opens anything.

A backend, an API format, a structured output method, a key source or the
orchestrator. The openers, the API formats, the structured output methods, the
key sources and the orchestrators register themselves, in packages the settings
may not import, so the settings cannot refuse an unknown name, nor a key of a
``key`` block that its source does not read. A component's opener would refuse
it, but only when that component opens, which can be after another model has
loaded; and the orchestrator would refuse it only once every component is open. So each
command that opens components calls ``check_registered_names`` while it reads
its settings, and reports a refusal as a usage error, as it reports any other
settings mistake. Once the ``key`` blocks pass, ``key_source_variables`` names
the environment variables their sources read keys from, for the deploy tool.

Example:
    >>> check_registered_names(WorkflowSettings())  # a use case's file: nothing raised
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..adapters.embedder.api.base_embedding_api_format import EMBEDDING_API_FORMATS
from ..adapters.key_source.base_key_source import KEY_SOURCES
from ..adapters.llm.api.base_generation_api_format import GENERATION_API_FORMATS
from ..adapters.llm.api.base_structured_output_method import STRUCTURED_OUTPUT_METHODS
from ..adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from ..adapters.reranker.api.base_rerank_api_format import RERANK_API_FORMATS
from ..config.components.component_settings import (
    ComponentSettings,
    LlmSettings,
    VectorStoreSettings,
)
from ..config.components.key_settings import KeySettings
from ..config.components.registered_component_settings import (
    RegisteredComponentSettings,
)
from ..config.workflow_settings import WorkflowSettings
from ..domain.registry import Registry
from .openers.component_openers import OPENERS_BY_TYPE
from .registered_components import registered_backend_class
from .standalone_key_source import checked_key_source

# The API formats of each model component type, by the type its settings name.
_API_FORMATS_BY_TYPE: dict[str, Registry[Any]] = {
    "embedder": EMBEDDING_API_FORMATS,
    "reranker": RERANK_API_FORMATS,
    "llm": GENERATION_API_FORMATS,
}
Place = tuple[str, ...]  # a value's path in the workflow file, its levels in order


def check_registered_names(settings: WorkflowSettings) -> None:
    """Refuse an unknown backend, API format, output method, key source or orchestrator.

    Every component is checked, whether the command opens it or not, the
    vector store included with the ``key`` block of its ``elasticsearch``
    block, every model's ``api`` block with its ``key`` block,
    and an LLM's ``structured_output`` in its ``api`` and ``served`` blocks,
    whatever the component's backend; then the orchestrator, whether the
    command runs the graph or not. A ``key`` block must hold the keys its
    source reads, each, and no other.

    Args:
        settings: The settings a command is about to open components from.

    Raises:
        ValueError: The first unknown name, with the variable it is read from,
            its sort and the known names: ``QP_COMPONENTS__LLM__API__API_FORMAT:
            unknown generation API format 'jina' (expected one of: 'openai')``.

    Example:
        >>> check_registered_names(WorkflowSettings(components={"llm": {"api": {
        ...     "api_format": "jina"}}}))
        ValueError: QP_COMPONENTS__LLM__API__API_FORMAT: unknown generation ...
    """
    variable_of = WorkflowSettings.variable_of
    for name, component in settings.components.items():
        backend = variable_of("components", name, "backend")
        _check(_known_backends(component), component.backend, backend)
    for name, model in settings.model_components().items():
        api_format = variable_of("components", name, "api", "api_format")
        _check(_API_FORMATS_BY_TYPE[model.type].get, model.api.api_format, api_format)
        if isinstance(model, LlmSettings):
            _check_structured_output(name, model)
    for place, block in _key_blocks(settings):
        checked_key_source(block, variable_of(*place))
    orchestrator = variable_of("orchestrator")
    _check(ORCHESTRATORS.get, settings.orchestrator, orchestrator)


def key_source_variables(settings: WorkflowSettings) -> dict[str, str]:
    """Name the environment variables the key sources of the settings read keys from.

    Every ``key`` block is read, whatever its component's backend, as the
    check of registered names reads them, which must have passed: the deploy
    tool refuses a value for each variable in its deploy file.

    Args:
        settings: The settings an App will read, their ``key`` blocks checked.

    Returns:
        Each variable, such as ``OPENAI_API_KEY`` for ``{source: env,
        variable: OPENAI_API_KEY}``, and the place of the block that names it
        in the workflow file, such as ``components.llm.api.key``.

    Example:
        >>> key_source_variables(WorkflowSettings())
        {'OPENAI_API_KEY': 'components.llm.api.key'}
    """
    return {
        variable: ".".join(place)
        for place, block in _key_blocks(settings)
        for variable in KEY_SOURCES.get(block.source).variables(block.read_keys())
    }


def _key_blocks(settings: WorkflowSettings) -> list[tuple[Place, KeySettings]]:
    # The store's in its elasticsearch block and every model's in its api block,
    # each with its place in the workflow file; a component without one has none.
    stores = [
        (("components", name, "elasticsearch", "key"), component.elasticsearch.key)
        for name, component in settings.components.items()
        if isinstance(component, VectorStoreSettings)
    ]
    models = [
        (("components", name, "api", "key"), model.api.key)
        for name, model in settings.model_components().items()
    ]
    return [(place, block) for place, block in stores + models if block is not None]


def _check_structured_output(name: str, llm: LlmSettings) -> None:
    # Both blocks, whatever the backend, as an api block's format is.
    for block_name, block in (("api", llm.api), ("served", llm.served)):
        variable = WorkflowSettings.variable_of(
            "components", name, block_name, "structured_output"
        )
        _check(STRUCTURED_OUTPUT_METHODS.get, block.structured_output, variable)


def _known_backends(component: ComponentSettings) -> Callable[[str], object]:
    # A type registered in COMPONENT_TYPES knows its backends by its classes,
    # served always among them; the toolkit's four by their openers.
    if isinstance(component, RegisteredComponentSettings):
        port = component.component_type()
        return lambda backend: registered_backend_class(port, backend)
    return OPENERS_BY_TYPE[component.type].get


def _check(find: Callable[[str], object], value: str, variable: str) -> None:
    try:
        find(value)  # raises, naming the known names, when value is unknown
    except ValueError as exc:
        raise ValueError(f"{variable}: {exc}") from exc
