"""A component type of the tests' own, ``shouter``, defined as a use case would.

A workflow file loads it by naming ``shout_extension`` under ``extensions``. It
imports the toolkit through ``extension_api`` alone, and no registry: the
component type and each implementation register on their class line.
"""

from __future__ import annotations

from pydantic import BaseModel

from query_processing.extension_api import (
    BaseComponent,
    ComponentResources,
    SettingsBlock,
)


class ShoutRequest(BaseModel):
    text: str


class Shouted(BaseModel):
    text: str


class BaseShouter(BaseComponent[ShoutRequest, Shouted], component_type="shouter"):
    """Says a text louder."""


class UpperShouterSettings(SettingsBlock):
    suffix: str = ""


class UpperShouter(BaseShouter, backend="local", settings=UpperShouterSettings):
    """Says the text in capitals, its suffix after it."""

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    @classmethod
    def from_settings(
        cls, settings: UpperShouterSettings, resources: ComponentResources
    ) -> UpperShouter:
        del resources  # a shouter needs nothing besides its own block
        return cls(settings.suffix)

    def invoke(self, request: ShoutRequest) -> Shouted:
        return Shouted(text=request.text.upper() + self.suffix)


class FakeShouter(BaseShouter, backend="fake"):
    """Says the text as it is: no settings, so no ``from_settings``."""

    def invoke(self, request: ShoutRequest) -> Shouted:
        return Shouted(text=request.text)
