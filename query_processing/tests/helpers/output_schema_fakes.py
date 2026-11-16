"""An output schema the toolkit's tests register, as a use case registers its own.

Importing the module registers it, once per run, under ``fixture_rewrite``: a
self-contained request and the conditions of a metadata filter, which a
``generate`` stage writes as two fields.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from query_processing.application.output_schemas import OUTPUT_SCHEMAS
from query_processing.domain.metadata_filter import MetadataCondition

FIXTURE_SCHEMA = "fixture_rewrite"


@OUTPUT_SCHEMAS.register(FIXTURE_SCHEMA)
class FixtureRewrite(BaseModel):
    """The user's message made self-contained, and the conditions it names."""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    metadata_filter: list[MetadataCondition]
