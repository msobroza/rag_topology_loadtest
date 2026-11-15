"""``rag-ingest``: a JSONL loader, and chunk, embed and upsert a corpus.

The synthetic handbook it ingests by default lives in ``domain/corpus.py``. It
embeds with the one embedder component of the workflow file, or with the one
``--embedder`` names when the file declares several.

Example:
    $ rag-ingest --synthetic 200          # embed with the RAG_* configured backend
    $ rag-ingest --corpus docs.jsonl --embedder embedder
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

if TYPE_CHECKING:  # the composition root is imported lazily, in _configured_composition
    from ..composition.composition_root import CompositionRoot


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
    parser.add_argument(
        "--embedder",
        metavar="NAME",
        help="the embedder component to embed with (default: the only one)",
    )
    return parser


async def _ingest_through(
    composition: CompositionRoot, passages: Sequence[Passage], args: argparse.Namespace
) -> dict[str, object]:
    batch_size = args.batch_size
    async with composition:
        embedder, store = await composition.ingest_ports(args.embedder)
        count = await ingest(passages, embedder, store, batch_size=batch_size)
        _, stored = await store.ready()
    return {"passages": count, "embedder": embedder.model_name, "store": stored}


def _configured_composition(
    parser: argparse.ArgumentParser, embedder: str | None
) -> CompositionRoot:
    from ..composition.composition_root import CompositionRoot
    from ..composition.registered_names import check_registered_names
    from ..config.rag_settings import RagSettings

    try:
        settings = RagSettings()
        check_registered_names(settings)
        settings.ingestion_embedder(embedder)
        store = settings.ingestion_store()
    except ValueError as exc:  # a settings mistake is a usage error, not a crash
        parser.error(str(exc))
    if settings.components[store].backend == "fake":
        backend = RagSettings.variable_of("components", store, "backend")
        raise SystemExit(
            f"rag-ingest: {backend}='fake' is seeded when the app starts and "
            "keeps nothing between processes; ingest needs 'elasticsearch'"
        )
    return CompositionRoot(settings)


def main(
    argv: Sequence[str] | None = None, *, composition: CompositionRoot | None = None
) -> int:
    """Run ``rag-ingest``: chunk, embed and upsert a corpus, then print one JSON line.

    The summary line is the ``ingest_done`` event: the document count, the passage
    count, the embedder's model name and what the store reports holding.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        composition: Builds the ports to ingest through; ``None`` builds one from
            the ``RAG_*`` settings, which refuses the ``fake`` vector store; a
            settings mistake, an unknown backend or API format, and an
            embedder or a vector store that cannot be told included, exits as
            a usage error.

    Returns:
        The exit status, 0.

    Example:
        >>> main(["--synthetic", "200"])                     # built from RAG_* settings
        >>> main(["--corpus", "docs.jsonl"], composition=CompositionRoot(settings))
    """
    parser = _parser()
    args = parser.parse_args(argv)
    composition = composition or _configured_composition(parser, args.embedder)
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
    summary = asyncio.run(_ingest_through(composition, passages, args))
    print(json.dumps({"event": "ingest_done", "documents": len(docs), **summary}))
    return 0
