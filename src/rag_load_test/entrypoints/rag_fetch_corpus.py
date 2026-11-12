"""``rag-fetch-corpus``: download a public corpus for ``rag-ingest`` and Locust.

It writes ``<out>/corpus.jsonl``, one ``{"id", "title", "text", "metadata"}``
object per line (``rag-ingest --corpus``), and ``<out>/questions.jsonl``, one
``{"question"}`` per line (``RAG_LOADTEST_QUESTIONS_FILE``). Run it on the machine
that ingests, not on a laptop: nothing of a corpus belongs in this repository.

Example:
    $ rag-fetch-corpus quebec-insurance --out data/quebec
    $ rag-ingest --corpus data/quebec/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from ..adapters.corpora.catalog import CORPORA, CorpusDefinition
from ..adapters.corpora.conversion import CorpusFormatError
from ..adapters.corpora.download import (
    CorpusDownloader,
    CorpusFetchError,
    FetchedCorpus,
)
from ..adapters.corpora.jsonl_files import write_jsonl_atomically
from ..adapters.shared.http_policy import HttpPolicy
from ..config.settings import RagSettings

CORPUS_FILE, QUESTIONS_FILE = "corpus.jsonl", "questions.jsonl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-fetch-corpus",
        description="Download a public corpus as rag-ingest and Locust read it.",
    )
    parser.add_argument("corpus", choices=sorted(CORPORA))
    parser.add_argument(
        "--out", type=Path, required=True, help="folder for the two JSONL files"
    )
    return parser


def _write(out: Path, fetched: FetchedCorpus) -> tuple[Path, Path]:
    # Both files after every download succeeded, each one renamed into place.
    out.mkdir(parents=True, exist_ok=True)
    corpus_file, questions_file = out / CORPUS_FILE, out / QUESTIONS_FILE
    write_jsonl_atomically(corpus_file, (d.model_dump() for d in fetched.documents))
    write_jsonl_atomically(questions_file, ({"question": q} for q in fetched.questions))
    return corpus_file, questions_file


def _fetch_and_write(
    corpus: CorpusDefinition, out: Path, downloader: CorpusDownloader
) -> dict[str, object]:
    # The summary line's fields.
    fetched = downloader.fetch(corpus)
    corpus_file, questions_file = _write(out, fetched)
    return {
        "event": "corpus_fetched",
        "corpus": corpus.name,
        "documents": len(fetched.documents),
        "questions": len(fetched.questions),
        "corpus_file": str(corpus_file),
        "questions_file": str(questions_file),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: RagSettings | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Run ``rag-fetch-corpus``: download, convert and write a corpus.

    Prints one JSON summary line (documents, questions, the two files), then the
    corpus's licence note. A network failure or a malformed line is reported on
    stderr, without a traceback, and nothing is written.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        settings: Source of the timeout, retries and backoff
            (``RAG_HTTP_TIMEOUT_S``, ``_RETRIES``, ``_BACKOFF_S``); ``None`` reads
            the ``RAG_*`` environment.
        transport: The HTTP transport; ``None`` uses the network. Tests pass an
            ``httpx.MockTransport``.

    Returns:
        The exit status: 0, or 1 after an expected failure.

    Example:
        >>> main(["quebec-insurance", "--out", "data/quebec"])
        0
    """
    args = _parser().parse_args(argv)
    settings = settings or RagSettings()
    policy = HttpPolicy(
        settings.http_timeout_s, settings.http_retries, settings.http_backoff_s
    )
    corpus = CORPORA[args.corpus]
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        try:
            summary = _fetch_and_write(
                corpus, args.out, CorpusDownloader(client, policy)
            )
        except (CorpusFetchError, CorpusFormatError, OSError) as exc:
            print(f"rag-fetch-corpus: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(summary, ensure_ascii=False))
    print(f"licence: {corpus.licence_note}")
    return 0
