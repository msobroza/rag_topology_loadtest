"""``rag-ingest``: a JSONL loader, and chunk, embed and upsert a corpus.

The synthetic handbook it ingests by default lives in ``domain/corpus.py``.

Example:
    $ rag-ingest --synthetic 200          # embed with the RAG_* configured backend
    $ rag-ingest --corpus docs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..application.ingest import ingest
from ..domain.corpus import SourceDocument, chunk, synthetic_corpus
from ..domain.passages import Passage

if TYPE_CHECKING:  # the composition root is imported lazily, in _configured_ports
    from ..adapters.rag_ports import RagPorts


def load_jsonl(path: Path) -> list[SourceDocument]:
    """Read a corpus: one ``{"id", "title", "text", "metadata"?}`` object per line.

    Blank lines are skipped.

    Args:
        path: The JSONL file to read.

    Returns:
        The documents, in file order.

    Raises:
        ValueError: If a line is not such an object; the message names the file
            and the line number.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return [
            _document(path, lineno, line)
            for lineno, line in enumerate(handle, start=1)
            if line.strip()
        ]


def _document(path: Path, lineno: int, line: str) -> SourceDocument:
    try:
        return SourceDocument.model_validate(json.loads(line))
    except ValueError as exc:  # json.JSONDecodeError and pydantic errors
        raise ValueError(f"{path}: line {lineno}: {exc}: {line[:80]!r}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ingest", description="Chunk, embed and upsert a corpus."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", type=int, default=200, help="number of synthetic documents"
    )
    source.add_argument("--corpus", type=Path, help="JSONL corpus file")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chunk-words", type=int, default=120)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


async def _ingest_through(
    ports: RagPorts, passages: Sequence[Passage], batch_size: int
) -> dict[str, object]:
    async with ports:
        embedder, store = await ports.ingest_ports()
        count = await ingest(passages, embedder, store, batch_size=batch_size)
        _, stored = await store.ready()
    return {"passages": count, "embedder": embedder.model_name, "store": stored}


def _configured_ports() -> RagPorts:
    from ..adapters.rag_ports import RagPorts
    from ..config.settings import RagSettings

    settings = RagSettings()
    if settings.vector_store_backend == "fake":
        raise SystemExit(
            "rag-ingest: RAG_VECTOR_STORE_BACKEND='fake' is seeded when the app starts "
            "and keeps nothing between processes; ingest needs 'elasticsearch'"
        )
    return RagPorts(settings)


def main(argv: Sequence[str] | None = None, *, ports: RagPorts | None = None) -> int:
    """Run ``rag-ingest``: chunk, embed and upsert a corpus, then print one JSON line.

    The summary line is the ``ingest_done`` event: the document count, the passage
    count, the embedder's model name and what the store reports holding.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        ports: The ports to ingest through; ``None`` builds them from the
            ``RAG_*`` settings, which refuses the ``fake`` vector store.

    Returns:
        The exit status, 0.

    Example:
        >>> main(["--synthetic", "200"])                     # ports from RAG_* settings
        >>> main(["--corpus", "docs.jsonl"], ports=RagPorts(settings))
    """
    args = _parser().parse_args(argv)
    ports = ports or _configured_ports()
    docs = (
        load_jsonl(args.corpus)
        if args.corpus
        else synthetic_corpus(args.synthetic, args.seed)
    )
    passages = [
        p
        for d in docs
        for p in chunk(
            d, chunk_words=args.chunk_words, overlap_words=args.overlap_words
        )
    ]
    summary = asyncio.run(_ingest_through(ports, passages, args.batch_size))
    print(json.dumps({"event": "ingest_done", "documents": len(docs), **summary}))
    return 0
