"""An unknown backend or API format, refused before a command opens anything.

The openers and the API formats register themselves, in packages the settings
may not import, so the settings cannot refuse an unknown name. A component's
opener would refuse it, but only when that component opens, which can be after
another model has loaded. So each command that opens components calls
``check_registered_names`` while it reads its settings, and reports a refusal
as a usage error, as it reports any other settings mistake.

Example:
    >>> check_registered_names(WorkflowSettings())  # a use case's file: nothing raised
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..adapters.embedder.api.base_embedding_api_format import EMBEDDING_API_FORMATS
from ..adapters.llm.api.base_generation_api_format import GENERATION_API_FORMATS
from ..adapters.reranker.api.base_rerank_api_format import RERANK_API_FORMATS
from ..config.component_settings import ComponentSettings
from ..config.registered_component_settings import RegisteredComponentSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.registry import Registry
from .openers.component_openers import OPENERS_BY_TYPE
from .registered_components import registered_backend_class

# The API formats of each model component type, by the type its settings name.
_API_FORMATS_BY_TYPE: dict[str, Registry[Any]] = {
    "embedder": EMBEDDING_API_FORMATS,
    "reranker": RERANK_API_FORMATS,
    "llm": GENERATION_API_FORMATS,
}


def check_registered_names(settings: WorkflowSettings) -> None:
    """Refuse a backend that has no opener, or an API format nobody registered.

    Every component is checked, whether the command opens it or not, the
    vector store included, and every model's ``api`` block, whatever the
    component's backend.

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
