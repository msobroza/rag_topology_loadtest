"""What a use case imports from the toolkit: every name it needs, from one module.

A use case holds the implementations that are its own (ADR 0006). Its workflow
file names its packages under ``extensions``; the toolkit imports every module
of each before it reads the components, and their classes register themselves.
A use case imports the toolkit through this module alone, and no package of
the toolkit imports it.

=======================  ==================================  =========================
To add                   Subclass                            Write
=======================  ==================================  =========================
An implementation of a   its abstract class, such as         ``invoke``, and
component type           ``BaseRouter``                      ``from_settings`` when it
                                                             has settings
A whole component type   ``BaseComponent[Request, Answer]``  its request and its answer
A stage type             ``BaseStage``                       what the stage does
                                                             beyond calling its
                                                             component
An edge condition        a function under the toolkit's      the function
                         decorator
An output schema         a pydantic model under the          its fields
                         toolkit's decorator
=======================  ==================================  =========================

Two spellings fill the same registries: a keyword on the class line for a
component, a decorator for the other entries (``STAGE_TYPES``,
``EDGE_CONDITIONS``, ``OUTPUT_SCHEMAS``, ``COMPONENT_API_FORMATS``).

Every component type has a ``served`` backend with no code: the toolkit builds
the class that calls a model server on its call route, and an API format that
mirrors its request and its answer. A use case redefines the format by
registering a ``BaseApiFormat`` under the type's name, and the served class by
naming ``backend="served"`` on a subclass of ``ServedCallMixin`` and the type.

A use case's tests import the test classes the toolkit ships from
``query_processing.testing``, not from here, so that no command loads what
they serve a component with: ``ComponentContract``, subclassed with
``make_component`` and ``make_request``, and ``RouterContract`` for a router,
which adds ``routes``.

Example:
    >>> from query_processing.extension_api import BaseComponent, SettingsBlock
    >>> class BaseShouter(BaseComponent[ShoutRequest, Shouted],
    ...                   component_type="shouter"):
    ...     '''Says a text louder.'''
    >>> class UpperShouter(BaseShouter, backend="local",
    ...                    settings=UpperShouterSettings):
    ...     def invoke(self, request: ShoutRequest) -> Shouted: ...
"""

from __future__ import annotations

from .adapters.router.base_router import BaseRouter
from .application.edge_conditions.edge_condition import EDGE_CONDITIONS
from .application.output_schemas import OUTPUT_SCHEMAS
from .application.stages.base_stage import (
    STAGE_TYPES,
    BaseStage,
    FieldTypes,
    StageContext,
    StageFields,
    StageSettings,
)
from .composition.composition_root import CompositionRoot
from .config.router_settings import RouterSettings
from .config.settings_blocks import SettingsBlock
from .config.workflow_settings import WorkflowSettings
from .domain.api_format import COMPONENT_API_FORMATS, BaseApiFormat
from .domain.base_component import BaseComponent
from .domain.component_requests import EmbedAnswer, EmbedRequest
from .domain.component_resources import ComponentResources
from .domain.errors import RagDependencyError
from .domain.routes import Route, RouteChoice, RouteRequest
from .infrastructure.endpoint.served_call import ServedCallMixin

__all__ = [
    "COMPONENT_API_FORMATS",
    "EDGE_CONDITIONS",
    "OUTPUT_SCHEMAS",
    "STAGE_TYPES",
    "BaseApiFormat",
    "BaseComponent",
    "BaseRouter",
    "BaseStage",
    "ComponentResources",
    "CompositionRoot",
    "EmbedAnswer",
    "EmbedRequest",
    "FieldTypes",
    "RagDependencyError",
    "Route",
    "RouteChoice",
    "RouteRequest",
    "RouterSettings",
    "ServedCallMixin",
    "SettingsBlock",
    "StageContext",
    "StageFields",
    "StageSettings",
    "WorkflowSettings",
]
