"""The blocks every component's settings are made of, whatever its type.

A block of the workflow file refuses a key it does not have. Every component
states its type and its backend (``ComponentSettings``), and may have an
``http`` block and a ``served`` block; the settings of each type are built on
these (``component_settings.py``, ``registered_component_settings.py``).

Example:
    >>> ComponentHttpSettings(retries=0).retries
    0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsBlock(BaseModel):
    """A block of the workflow file: frozen, and refusing a key it does not have.

    Example:
        >>> FakeStoreSettings.model_validate({"seed_document": 3})
        ValidationError: ... seed_document ... Extra inputs are not permitted
    """

    # A key the shape does not have is a typo, never a silent default.
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServedSettings(SettingsBlock):
    """A ``served`` block: the full route of the model server of ours that serves it.

    The rest of the endpoint follows from the component, so it is not set: the
    standard format of its type, its alias as the model name, and the Domino
    run token when the app runs on Domino (``composition/http_resources.py``).

    Example:
        >>> ServedSettings(url="http://reranker-app/v3/rerank").url
        'http://reranker-app/v3/rerank'
    """

    url: str = ""  # the full route, e.g. http://host:8001/v3/rerank


class ComponentHttpSettings(SettingsBlock):
    """A component's ``http`` block: how patiently it is called over HTTP.

    It applies on ``served`` and on ``api`` alike, so it is the component's,
    not one backend's: a topology that moves a component from ``local`` to
    ``served`` keeps it. Each value it leaves out is the default every
    component shares, the workflow file's top-level ``http`` block
    (``QP_HTTP__TIMEOUT_S``, ``QP_HTTP__RETRIES``, ``QP_HTTP__BACKOFF_S``);
    ``composition/http_resources.py`` resolves them.

    Example:
        >>> ComponentHttpSettings(timeout_s=120).model_dump(exclude_none=True)
        {'timeout_s': 120.0}
    """

    timeout_s: float | None = Field(default=None, gt=0)  # of one attempt
    retries: int | None = Field(default=None, ge=0)  # attempts after the first
    # The wait before the first retry, doubled before each next one.
    backoff_s: float | None = Field(default=None, ge=0)


class ComponentSettings(SettingsBlock):
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
