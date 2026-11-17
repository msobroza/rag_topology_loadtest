"""The target record: what a run's target said about itself, beside its statistics.

A run writes ``<csv prefix>_target.json`` as it stops: its target, and each
topology alias and run label the target reported in its headers, with how many
responses carried each. A workflow app given no topology alias sends no
``X-RAG-Topology`` header, and its topology is recorded as ``null``, never as a
name nobody chose. When any response reported a stage's tokens, the record
also holds ``tokens``: each such stage, named as its STAGE row is
(``/query:generate``, ``/retrieve:rewrite``), with its totals over the run.
``qp-compare`` reads the record back through the same class, so a record that
is not of this shape fails with a message naming the file. A record written
before tokens were counted has no ``tokens`` key, and still reads; one main's
load test wrote, whose ``tokens`` is one total of ``/query`` answers with the
time of their generate stage as ``generate_ms``, reads as the one stage
``/query:generate``.

Example:
    >>> write_target_record("results/monolith", "rag", Counter({("monolith", "m"): 3}))
    >>> read_target_record(Path("results/monolith_target.json")).seen[0].run_label
    'm'
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .loadtest_settings import LoadTarget
from .response_checks import SpentTokens

RECORD_SUFFIX = "_target.json"  # after Locust's --csv prefix
RECORD_SHAPE = (
    '{"target": "rag" | "rerank" | "embeddings", '
    '"seen": [{"topology": str | null, "run_label": str, "responses": int >= 0}], '
    '"tokens"?: {<STAGE row name>: {"responses": int >= 0, "prompt": int >= 0, '
    '"completion": int >= 0, "stage_ms": float >= 0}}}'
)
# The stage main's load test counted the tokens of (pull request 27): the
# generate stage of /query answers, named as its STAGE row is.
MAIN_TOKENS_STAGE = "/query:generate"
# The time key of that record, which no per-stage record writes: it tells the
# two shapes apart.
MAIN_TIME_KEY = "generate_ms"
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


class StageTokenTotals(BaseModel):
    """One stage's tokens over a run, with the stage's time over the same responses.

    Example:
        >>> StageTokenTotals(responses=0, prompt=0, completion=0, stage_ms=0.0).plus(
        ...     SpentTokens(829, 128, 7474.0)).completion
        128
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    responses: int = Field(ge=0)
    prompt: int = Field(ge=0)
    completion: int = Field(ge=0)
    stage_ms: float = Field(ge=0)

    def plus(self, spent: SpentTokens) -> StageTokenTotals:
        """Return these totals with one more response of the stage counted.

        Args:
            spent: The tokens the stage spent in that response, and its time.

        Returns:
            New totals; these are left as they are.
        """
        return StageTokenTotals(
            responses=self.responses + 1,
            prompt=self.prompt + spent.prompt,
            completion=self.completion + spent.completion,
            stage_ms=self.stage_ms + spent.stage_ms,
        )


NO_TOKENS = StageTokenTotals(responses=0, prompt=0, completion=0, stage_ms=0.0)


def count_spent_tokens(
    totals: dict[str, StageTokenTotals], spent: Iterable[tuple[str, SpentTokens]]
) -> None:
    """Add one response's tokens to the run's totals, each under its stage.

    Args:
        totals: The run's totals by STAGE row name; changed in place.
        spent: The response's ``(STAGE row name, SpentTokens)`` pairs.

    Example:
        >>> totals: dict[str, StageTokenTotals] = {}
        >>> spent = SpentTokens(829, 128, 7474.0)
        >>> count_spent_tokens(totals, [("/query:generate", spent)])
        >>> totals["/query:generate"].responses
        1
    """
    for stage, tokens in spent:
        totals[stage] = totals.get(stage, NO_TOKENS).plus(tokens)


class TargetRecord(BaseModel):
    """What one run observed its target reporting; ``seen`` is most responses first.

    ``tokens`` maps a STAGE row name to that stage's totals; None when no
    response of the run reported a stage's tokens.
    """

    model_config = ConfigDict(extra="forbid")

    target: LoadTarget
    seen: list[TargetSighting]
    tokens: dict[str, StageTokenTotals] | None = None

    @field_validator("tokens", mode="before")
    @classmethod
    def _read_mains_tokens(cls, tokens: object) -> object:
        # main's record holds one total of /query answers, its time under
        # generate_ms; it reads as that one stage, any other key of it refused.
        if not isinstance(tokens, dict) or MAIN_TIME_KEY not in tokens:
            return tokens
        renamed = {"answers": "responses", MAIN_TIME_KEY: "stage_ms"}
        return {MAIN_TOKENS_STAGE: {renamed.get(k, k): v for k, v in tokens.items()}}


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
    tokens: Mapping[str, StageTokenTotals] | None = None,
) -> None:
    """Write ``<csv_prefix>_target.json``, most responses first.

    Nothing is written without ``--csv``. Model servers send no topology headers,
    so their ``seen`` is empty: it is left empty for any target but ``rag``.

    Args:
        csv_prefix: Locust's ``--csv`` prefix; ``None`` or ``""`` writes nothing.
        target: The run's ``QP_LOADTEST_TARGET``.
        counts: Responses per ``(topology alias, run label)``; ties are ordered
            by the pair, no topology alias first.
        tokens: The run's token totals by STAGE row name, written in the
            order of the names; a stage that counts no response is left out,
            and the record has no ``tokens`` key when none is left.

    Example:
        >>> write_target_record("results/monolith", "rag", Counter({("m", "m"): 3}))
    """
    if not csv_prefix:
        return
    seen = sightings_of(counts) if target == "rag" else []
    counted = {
        stage: totals
        for stage, totals in sorted((tokens or {}).items())
        if totals.responses > 0
    }
    record = TargetRecord(target=target, seen=seen, tokens=counted or None)
    # Only the absent total is left out: a None topology alias is written.
    left_out = set() if counted else {"tokens"}
    path = Path(f"{csv_prefix}{RECORD_SUFFIX}")
    path.write_text(json.dumps(record.model_dump(exclude=left_out), indent=2))


def sightings_of(counts: Mapping[TargetPair, int]) -> list[TargetSighting]:
    """Return the pairs a target reported, most responses first.

    Args:
        counts: Responses per ``(topology alias, run label)``; ties are ordered
            by the pair, no topology alias first.

    Returns:
        One sighting per pair.

    Example:
        >>> sightings_of({(None, "m"): 1, ("a", "m"): 3})[0].topology
        'a'
    """
    ordered = sorted(counts.items(), key=_most_responses_first)
    return [
        TargetSighting(topology=topology, run_label=label, responses=n)
        for (topology, label), n in ordered
    ]


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
