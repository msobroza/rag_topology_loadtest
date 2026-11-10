"""Corpus tooling: a seeded synthetic handbook, a JSONL loader, chunking, ``rag-ingest``.

The synthetic corpus describes a fictional company, Northwind Analytics. Each
topic owns a distinctive vocabulary that its sentences repeat and its question
reuses, so retrieval stays meaningful even with the bag-of-words ``FakeEmbedder``.

Example::

    rag-ingest --synthetic 200          # embed with the RAG_* configured backend
    rag-ingest --corpus docs.jsonl --fake
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field

from .models import EmbedderPort, MetadataValue, Passage, VectorStorePort


class SourceDocument(BaseModel):
    id: str
    title: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class Topic(NamedTuple):
    title: str
    question: str
    sentences: tuple[str, ...]


# Numeric slots ({n}, {days}, {amount}, {hours}) are filled from the seeded RNG so
# documents of the same topic differ.
TOPICS: dict[str, Topic] = {
    "vacation-policy": Topic(
        "Northwind Analytics Vacation Policy (revision {n})",
        "How many paid vacation days do employees accrue each year at Northwind Analytics?",
        (
            "Full-time employees at Northwind Analytics accrue {days} paid vacation days each year; part-time employees accrue vacation pro rata.",
            "Vacation days must be requested at least {n} days ahead so the team can plan cover for the vacation.",
            "Employees may carry up to {n} unused paid vacation days into the next year; further vacation days are forfeited.",
        ),
    ),
    "expense-reimbursement": Topic(
        "Northwind Analytics Expense Reimbursement Guide (revision {n})",
        "Do I need a receipt for an expense claim under {amount} dollars to get reimbursement at Northwind Analytics?",
        (
            "Northwind Analytics pays reimbursement for a business expense claim filed through the claim portal within {n} weeks.",
            "An itemized receipt is required for a claim over {amount} dollars; a claim under {amount} dollars needs only a short note instead of a receipt.",
            "Reimbursement lands with the next payroll after the manager approves the expense claim; alcohol is never a reimbursable expense.",
        ),
    ),
    "remote-work": Topic(
        "Northwind Analytics Remote Work Policy (revision {n})",
        "How often can employees work remotely from home each week at Northwind Analytics?",
        (
            "Employees at Northwind Analytics may work remotely from home up to {n} times per week with their manager's agreement.",
            "Remote work from home requires a secure home network and a company computer; personal devices are blocked.",
            "Each team picks one shared office day per week; employees who work remotely still join the office {n} weeks per quarter.",
        ),
    ),
    "security-training": Topic(
        "Northwind Analytics Security Training Requirements (revision {n})",
        "How often do staff finish the mandatory security module and phishing tests at Northwind Analytics?",
        (
            "Every quarter all staff at Northwind Analytics finish the mandatory security module within {n} weeks of it opening.",
            "The security module covers phishing, password hygiene and safe file handling and takes about {hours} hours.",
            "Staff who fail a monthly phishing test finish an extra security module within {hours} hours of the phishing test results.",
        ),
    ),
    "onboarding": Topic(
        "Northwind Analytics Onboarding Checklist (revision {n})",
        "What does the onboarding checklist cover for a new hire and their buddy at Northwind Analytics?",
        (
            "Onboarding at Northwind Analytics lasts {n} weeks and follows an onboarding checklist owned by the hire's manager.",
            "The onboarding checklist covers accounts, badge access, tooling setup and {n} introductory meetings for the hire.",
            "Every new hire is paired with an onboarding buddy within {hours} hours; the buddy stays assigned for {n} weeks.",
        ),
    ),
    "incident-response": Topic(
        "Northwind Analytics Incident Response Runbook (revision {n})",
        "Within how many minutes must the on-call engineers respond to a production incident alert at Northwind Analytics?",
        (
            "The on-call engineers must respond to a production incident alert within {n} minutes of the alert firing.",
            "A severity-one production incident pages {n} engineers and the on-call lead coordinates the fix.",
            "A blameless incident write-up is published within {hours} hours of the outage and shared with all engineers.",
        ),
    ),
    "data-retention": Topic(
        "Northwind Analytics Data Retention Policy (revision {n})",
        "How long does Northwind Analytics retain customer data and backups?",
        (
            "The data retention policy at Northwind Analytics is to retain customer data for {n} months after a contract ends.",
            "Encrypted backups of customer data are retained for {n} months, then the backups are rotated and destroyed.",
            "Customers may ask to have their customer data deleted early; the retention team removes the data and its backups within {n} weeks.",
        ),
    ),
    "travel-booking": Topic(
        "Northwind Analytics Travel Booking Policy (revision {n})",
        "Do I need approval for a business trip flight before the airfare is booked at Northwind Analytics?",
        (
            "Staff get approval for every business trip via the Northwind Analytics travel portal at least {n} weeks before the flight departs.",
            "A flight over {amount} dollars requires manager approval via the portal before the airfare is booked.",
            "Hotel rates are capped at {amount} dollars per night for a business trip unless the travel desk documents an exception.",
        ),
    ),
    "performance-reviews": Topic(
        "Northwind Analytics Performance Review Process (revision {n})",
        "How often is a performance review held and how is the review score scale applied at Northwind Analytics?",
        (
            "A performance review is held twice a year at Northwind Analytics; a self-evaluation is due on day {n} of the review cycle.",
            "Each performance review produces a review score on a scale from 1 to {n}, backed by written examples.",
            "Peer evaluation from at least {n} colleagues is gathered before the performance review score is decided.",
        ),
    ),
    "equipment-requests": Topic(
        "Northwind Analytics Equipment Request Procedure (revision {n})",
        "Where do I request a laptop or monitor from IT at Northwind Analytics, and when does the hardware arrive?",
        (
            "Staff request a laptop, monitor or headset from IT through the Northwind Analytics hardware portal; IT answers each hardware request within {hours} hours.",
            "A standard laptop request from IT is approved automatically and the laptop arrives from the IT stockroom within {n} weeks.",
            "A hardware request above a {amount} dollar budget needs a written justification before IT orders the equipment.",
        ),
    ),
}


def _slots(rng: random.Random) -> dict[str, int]:
    return {
        "n": rng.randint(1, 30),
        "days": rng.randint(5, 30),
        "amount": 25 * rng.randint(1, 100),
        "hours": rng.choice((2, 4, 8, 24, 48, 72)),
    }


def synthetic_corpus(n: int, seed: int = 1234) -> list[SourceDocument]:
    """``n`` deterministic handbook documents cycling through ``TOPICS``.

    Example::

        synthetic_corpus(3)[0].metadata   # {"topic": "vacation-policy", "seq": 0}
    """
    rng = random.Random(seed)
    slugs = list(TOPICS)
    docs = []
    for i in range(n):
        slug = slugs[i % len(slugs)]
        topic = TOPICS[slug]
        docs.append(
            SourceDocument(
                id=f"doc-{i:04d}",
                title=topic.title.format_map(_slots(rng)),
                text=" ".join(s.format_map(_slots(rng)) for s in topic.sentences),
                metadata={"topic": slug, "seq": i},
            )
        )
    return docs


def synthetic_questions(n: int = 50, seed: int = 1234) -> list[str]:
    """``n`` questions cycling through the topics, matching ``synthetic_corpus``."""
    rng = random.Random(seed)
    topics = list(TOPICS.values())
    return [topics[i % len(topics)].question.format_map(_slots(rng)) for i in range(n)]


def load_jsonl(path: Path) -> list[SourceDocument]:
    """One ``{"id", "title", "text", "metadata"?}`` object per non-blank line."""
    docs = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                docs.append(SourceDocument.model_validate(json.loads(line)))
            except ValueError as exc:  # json.JSONDecodeError and pydantic errors
                raise ValueError(
                    f"{path}: line {lineno}: {exc}: {line[:80]!r}"
                ) from exc
    return docs


def chunk(
    doc: SourceDocument, *, chunk_words: int = 120, overlap_words: int = 20
) -> list[Passage]:
    """Sliding word windows; ids are ``<doc id>:<chunk index>``.

    Example::

        chunk(doc, chunk_words=100, overlap_words=20)[1].id   # "doc-0001:1"
    """
    if not 0 <= overlap_words < chunk_words:
        raise ValueError(
            f"need 0 <= overlap_words < chunk_words, got {overlap_words}, {chunk_words}"
        )
    words = doc.text.split()
    step = chunk_words - overlap_words
    starts = range(0, max(len(words) - overlap_words, 1), step) if words else []
    return [
        Passage(
            id=f"{doc.id}:{i}",
            text=" ".join(words[start : start + chunk_words]),
            metadata={**doc.metadata, "doc_id": doc.id, "title": doc.title, "chunk": i},
        )
        for i, start in enumerate(starts)
    ]


async def ingest(
    passages: Sequence[Passage],
    embedder: EmbedderPort,
    store: VectorStorePort,
    *,
    batch_size: int = 64,
) -> int:
    """Embed passages in batches and upsert them; returns the number upserted."""
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        await store.upsert(batch, await embedder.embed([p.text for p in batch]))
    return len(passages)


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
        "--fake", action="store_true", help="FakeEmbedder + in-memory store (demo)"
    )
    return parser


async def _ingest_configured(
    passages: Sequence[Passage], batch_size: int
) -> dict[str, object]:
    import httpx

    from .adapters import INDEX, model_executor
    from .ovms import TIMEOUT_S
    from .settings import RagSettings
    from .workflow import build_embedder, build_vector_store

    settings = RagSettings()
    executor = model_executor()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        embedder = build_embedder(settings, client=client, executor=executor)
        store = build_vector_store(settings)
        count = await ingest(passages, embedder, store, batch_size=batch_size)
    executor.shutdown(wait=True)
    return {"passages": count, "index": INDEX, "embedder": embedder.model_name}


async def _ingest_fake(
    passages: Sequence[Passage], batch_size: int
) -> dict[str, object]:
    from .fakes import FakeEmbedder, InMemoryVectorStore

    count = await ingest(
        passages, FakeEmbedder(), InMemoryVectorStore(), batch_size=batch_size
    )
    return {"passages": count, "index": "in-memory", "embedder": "fake-embedder"}


def main(argv: Sequence[str] | None = None) -> int:
    """``rag-ingest``: prints one JSON summary line."""
    args = _parser().parse_args(argv)
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
    run = _ingest_fake if args.fake else _ingest_configured
    summary = asyncio.run(run(passages, args.batch_size))
    print(json.dumps({"event": "ingest_done", "documents": len(docs), **summary}))
    return 0
