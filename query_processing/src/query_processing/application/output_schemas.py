"""``OUTPUT_SCHEMAS``: the pydantic models a ``generate`` stage can ask the LLM for.

A stage names one by ``output_schema``; the LLM's answer is then one instance
of it, and the stage writes each of its fields to the field of ``writes`` of
the same name (spec 5). A use case registers its own schema with the
decorator, in a module of its extension package.

Example:
    >>> @OUTPUT_SCHEMAS.register("verdict")
    ... class Verdict(BaseModel):
    ...     model_config = ConfigDict(extra="forbid")
    ...     ok: bool
    >>> OUTPUT_SCHEMAS.get("verdict")
    <class 'Verdict'>
"""

from __future__ import annotations

from pydantic import BaseModel

from ..domain.registry import Registry

OUTPUT_SCHEMAS: Registry[type[BaseModel]] = Registry("output schema")
