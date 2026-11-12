"""The target record: what a run's target said about itself, beside its statistics.

A run writes ``<csv prefix>_target.json`` as it stops: its target, and each
topology and run label the target reported in its headers, with how many
responses carried each. ``rag-compare`` reads it back through the same class, so
a record that is not of this shape fails with a message naming the file.

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

RECORD_SUFFIX = "_target.json"  # after Locust's --csv prefix
RECORD_SHAPE = (
    '{"target": "rag" | "rerank" | "embeddings", '
    '"seen": [{"topology": str, "run_label": str, "responses": int >= 0}]}'
)


class TargetSighting(BaseModel):
    """One topology and run label the target reported, and how many responses did."""

    model_config = ConfigDict(extra="forbid")

    topology: str
    run_label: str
    responses: int = Field(ge=0)


class TargetRecord(BaseModel):
    """What one run observed its target reporting; ``seen`` is most responses first."""

    model_config = ConfigDict(extra="forbid")

    target: LoadTarget
    seen: list[TargetSighting]


def target_pair(headers: Mapping[str, str] | None) -> tuple[str, str] | None:
    """Return the ``(topology, run label)`` a response reported in its headers.

    A response that reported neither is ``None``: a proxy's error page or a
    connection error is not the target speaking, and counting it would make one
    configuration look like two.

    Args:
        headers: The response headers (``X-RAG-Topology``, ``X-RAG-Run-Label``);
            ``None`` when there were none.

    Returns:
        The pair, ``""`` for the one header missing, or ``None`` when both are.

    Example:
        >>> target_pair({"X-RAG-Topology": "monolith"})
        ('monolith', '')
    """
    headers = headers or {}
    topology, run_label = headers.get("X-RAG-Topology"), headers.get("X-RAG-Run-Label")
    if topology is None and run_label is None:
        return None
    return topology or "", run_label or ""


def write_target_record(
    csv_prefix: str | None, target: LoadTarget, counts: Mapping[tuple[str, str], int]
) -> None:
    """Write ``<csv_prefix>_target.json``, most responses first.

    Nothing is written without ``--csv``. Model servers send no topology headers,
    so their ``seen`` is empty: it is left empty for any target but ``rag``.

    Args:
        csv_prefix: Locust's ``--csv`` prefix; ``None`` or ``""`` writes nothing.
        target: The run's ``RAG_LOADTEST_TARGET``.
        counts: Responses per ``(topology, run label)``; ties are ordered by the
            pair.

    Example:
        >>> write_target_record("results/monolith", "rag", Counter({("m", "m"): 3}))
    """
    if not csv_prefix:
        return
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    seen = [
        TargetSighting(topology=topology, run_label=label, responses=n)
        for (topology, label), n in ordered
        if target == "rag"
    ]
    record = TargetRecord(target=target, seen=seen)
    path = Path(f"{csv_prefix}{RECORD_SUFFIX}")
    path.write_text(json.dumps(record.model_dump(), indent=2))


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
