"""The target record: what a run's target said about itself, beside its statistics.

A run writes ``<csv prefix>_target.json`` as it stops: its target, and each
topology alias and run label the target reported in its headers, with how many
responses carried each. A workflow app given no topology alias sends no
``X-RAG-Topology`` header, and its topology is recorded as ``null``, never as a
name nobody chose. When any answer reported its tokens, the record also holds
their total. ``qp-compare`` reads the record back through the same class, so a
record that is not of this shape fails with a message naming the file. A record
written before tokens were counted has no ``tokens`` key, and still reads.

Example:
    >>> write_target_record("results/monolith", "rag", Counter({("monolith", "m"): 3}))
    >>> read_target_record(Path("results/monolith_target.json")).seen[0].run_label
    'm'
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .loadtest_settings import LoadTarget
from .response_checks import AnswerTokens

RECORD_SUFFIX = "_target.json"  # after Locust's --csv prefix
RECORD_SHAPE = (
    '{"target": "rag" | "rerank" | "embeddings", '
    '"seen": [{"topology": str | null, "run_label": str, "responses": int >= 0}], '
    '"tokens"?: {"answers": int >= 0, "prompt": int >= 0, "completion": int >= 0, '
    '"llm_ms": float >= 0}}'
)
# What one response reported: its topology alias (None without the header) and
# its run label.
TargetPair = tuple[str | None, str]


class TargetSighting(BaseModel):
    """One topology alias and run label the target reported, and how many did.

    ``topology`` is None for responses that carried no ``X-RAG-Topology``.
    """

    model_config = ConfigDict(extra="forbid")

    topology: str | None
    run_label: str
    responses: int = Field(ge=0)


class TokenTotals(BaseModel):
    """The tokens of a run's answers, added up, with the time the LLM took on them.

    Example:
        >>> TokenTotals(answers=0, prompt=0, completion=0, llm_ms=0.0).plus(
        ...     AnswerTokens(829, 128, 7474.0)).completion
        128
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: int = Field(ge=0)
    prompt: int = Field(ge=0)
    completion: int = Field(ge=0)
    llm_ms: float = Field(ge=0)

    def plus(self, answer: AnswerTokens) -> TokenTotals:
        """Return these totals with one more answer counted.

        Args:
            answer: The answer's token counts and the time of its stages.

        Returns:
            New totals; these are left as they are.
        """
        return TokenTotals(
            answers=self.answers + 1,
            prompt=self.prompt + answer.prompt,
            completion=self.completion + answer.completion,
            llm_ms=self.llm_ms + answer.llm_ms,
        )


class TargetRecord(BaseModel):
    """What one run observed its target reporting; ``seen`` is most responses first.

    ``tokens`` is None when no answer of the run reported its tokens.
    """

    model_config = ConfigDict(extra="forbid")

    target: LoadTarget
    seen: list[TargetSighting]
    tokens: TokenTotals | None = None


def target_pair(headers: Mapping[str, str] | None) -> TargetPair | None:
    """Return the ``(topology alias, run label)`` a response reported in its headers.

    A response that reported neither is ``None``: a proxy's error page or a
    connection error is not the target speaking, and counting it would make one
    configuration look like two. A workflow app given no topology alias sends
    the run label alone.

    Args:
        headers: The response headers (``X-RAG-Topology``, ``X-RAG-Run-Label``);
            ``None`` when there were none.

    Returns:
        The pair, the topology alias None without its header and the run label
        ``""`` without its own, or ``None`` when both are missing.

    Example:
        >>> target_pair({"X-RAG-Run-Label": "embedder=local+reranker=local"})
        (None, 'embedder=local+reranker=local')
    """
    headers = headers or {}
    topology, run_label = headers.get("X-RAG-Topology"), headers.get("X-RAG-Run-Label")
    if topology is None and run_label is None:
        return None
    return topology, run_label or ""


def write_target_record(
    csv_prefix: str | None,
    target: LoadTarget,
    counts: Mapping[TargetPair, int],
    tokens: TokenTotals | None = None,
) -> None:
    """Write ``<csv_prefix>_target.json``, most responses first.

    Nothing is written without ``--csv``. Model servers send no topology headers,
    so their ``seen`` is empty: it is left empty for any target but ``rag``.

    Args:
        csv_prefix: Locust's ``--csv`` prefix; ``None`` or ``""`` writes nothing.
        target: The run's ``QP_LOADTEST_TARGET``.
        counts: Responses per ``(topology alias, run label)``; ties are ordered
            by the pair, no topology alias first.
        tokens: The run's token totals; the record has no ``tokens`` key when
            they are None or count no answer, never a total of zeros.

    Example:
        >>> write_target_record("results/monolith", "rag", Counter({("m", "m"): 3}))
    """
    if not csv_prefix:
        return
    ordered = sorted(counts.items(), key=_most_responses_first)
    seen = [
        TargetSighting(topology=topology, run_label=label, responses=n)
        for (topology, label), n in ordered
        if target == "rag"
    ]
    counted = tokens if tokens is not None and tokens.answers > 0 else None
    record = TargetRecord(target=target, seen=seen, tokens=counted)
    # Only the absent total is left out: a None topology alias is written.
    left_out = {"tokens"} if counted is None else set()
    path = Path(f"{csv_prefix}{RECORD_SUFFIX}")
    path.write_text(json.dumps(record.model_dump(exclude=left_out), indent=2))


def _most_responses_first(item: tuple[TargetPair, int]) -> tuple[int, str, str]:
    # None cannot be compared with a name, so no topology alias sorts as "".
    (topology, run_label), responses = item
    return -responses, topology or "", run_label


def read_target_record(path: Path) -> TargetRecord | None:
    """Read a target record, or ``None`` when the run left none.

    Args:
        path: The ``<csv prefix>_target.json`` file.

    Returns:
        The record, or ``None`` when the file does not exist.

    Raises:
        ValueError: If the file is not a target record; the message names the
            file, the field at fault and the shape expected.

    Example:
        >>> read_target_record(Path("results/monolith_target.json")).target
        'rag'
    """
    if not path.exists():
        return None
    try:
        return TargetRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "the document"
        raise ValueError(
            f"{path}: {field}: {first['msg']}; expected a target record "
            f"{RECORD_SHAPE}"
        ) from exc
