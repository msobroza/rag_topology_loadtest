"""A component type of the tests' own, ``grader``, whose answer a label edge reads.

Its answer's ``level`` is one of two labels, which ``answer_labels`` declares:
a plain call gives them to the field it writes ``level`` to, and a conditional
edge on ``label`` that reads that field has one row of ``path_map`` for each.
"""

from __future__ import annotations

from pydantic import BaseModel

from query_processing.extension_api import BaseComponent

from .shouting import ShoutRequest


class Grade(BaseModel):
    level: str


class BaseGrader(BaseComponent[ShoutRequest, Grade], component_type="grader"):
    """Grades how loud a text is: quiet or loud."""

    @classmethod
    def answer_labels(
        cls, type_settings: BaseModel | None
    ) -> dict[str, tuple[str, ...]]:
        del type_settings  # every grader answers one of the same two labels
        return {"level": ("quiet", "loud")}


class CapitalsGrader(BaseGrader, backend="local"):
    """Loud when the text is in capitals."""

    def invoke(self, request: ShoutRequest) -> Grade:
        return Grade(level="loud" if request.text.isupper() else "quiet")
