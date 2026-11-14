"""Write JSON lines atomically: a temporary file in the same folder, then a rename.

An interrupted run leaves the previous file, or none, never half of one; a run
that completes replaces what was there.

Example:
    >>> write_jsonl_atomically(Path("out/questions.jsonl"), [{"question": "Qui ?"}])
    1
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO


def write_jsonl_atomically(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Write one JSON object per line to ``path``, all or nothing.

    Args:
        path: The file to write; its folder must exist.
        rows: The objects, written in order, non-ASCII text kept as is.

    Returns:
        How many lines were written.

    Example:
        >>> write_jsonl_atomically(out / "corpus.jsonl", (d.model_dump() for d in docs))
        1223
    """
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed before the rename
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            count = _write_rows(handle, rows)
        os.replace(handle.name, path)  # atomic within one file system
    except BaseException:  # KeyboardInterrupt too: never leave the temporary behind
        Path(handle.name).unlink(missing_ok=True)
        raise
    return count


def _write_rows(handle: IO[str], rows: Iterable[Mapping[str, object]]) -> int:
    # Counted here rather than with len(): rows may be a generator.
    count = 0
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
    return count
