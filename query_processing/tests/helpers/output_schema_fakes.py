"""An output schema the toolkit's tests register, as a use case registers its own.

Importing the module registers it, once per run, under ``fixture_rewrite``: a
self-contained request and the conditions of a metadata filter, which a
``generate`` stage writes as two fields. ``ScriptedLLM`` answers a text it is
given and records what it was asked.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from query_processing_core.retrieval.metadata_filter import MetadataCondition

from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.application.output_schemas import OUTPUT_SCHEMAS
from query_processing.domain.components.component_requests import GenerateAnswer

FIXTURE_SCHEMA = "fixture_rewrite"


@OUTPUT_SCHEMAS.register(FIXTURE_SCHEMA)
class FixtureRewrite(BaseModel):
    """The user's message made self-contained, and the conditions it names."""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    metadata_filter: list[MetadataCondition]


class ScriptedLLM(BaseLLM):
    """Answers one text, and records the messages and schema of each call."""

    model_name = "scripted-llm"

    def __init__(self, answer: str) -> None:
        self.answer, self.schemas = answer, list[type[BaseModel] | None]()
        self.messages = list[list[dict[str, str]]]()

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        self.schemas.append(output_schema)
        self.messages.append(messages)
        return GenerateAnswer(text=self.answer)
