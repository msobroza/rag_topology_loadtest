"""What a load-test user sends: the request pool, and one plan per request.

The questions come from ``RAG_LOADTEST_QUESTIONS_FILE`` when it is set (what
``rag-fetch-corpus`` writes, or plain text), else they are synthetic; a request
to a model server carries what the workflow app's stage would send. Values in,
plans out: ``locustfile.py`` only does the Locust I/O around them, so they are
tested with plain calls and without gevent monkey-patching the interpreter.

Example:
    >>> pool = build_request_pool(settings.question_count, settings.top_k_retrieve,
    ...                           settings.top_k_rerank,
    ...                           questions_file=settings.questions_file)
    >>> plan = plan_rag_request("/query", pool, rng)
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..domain.api_endpoint import MODEL_SERVER_EMBEDDINGS_PATH, MODEL_SERVER_RERANK_PATH
from ..domain.corpus import chunk, synthetic_corpus, synthetic_questions
from ..domain.standard_formats import CohereRerankRequest, OpenAIEmbeddingsRequest

QUESTIONS_FILE_VAR = "RAG_LOADTEST_QUESTIONS_FILE"  # named in the reader's errors


@dataclass(frozen=True)
class RequestPool:
    """What requests draw from: the questions, the rerank candidates and ``top_n``.

    The candidates and ``top_n`` are what the workflow's rerank stage would send
    (``RAG_TOP_K_RETRIEVE`` / ``_RERANK``).
    """

    questions: Sequence[str]
    rerank_documents: Sequence[str]
    top_n: int


def build_request_pool(
    question_count: int,
    candidates: int,
    top_n: int,
    *,
    questions_file: Path | None = None,
) -> RequestPool:
    """Build the pool: the questions, and the first ``candidates`` passages.

    The questions are every one of ``questions_file`` when it is given, else
    ``question_count`` synthetic ones. The rerank and embeddings targets draw
    their queries from the same questions; the rerank candidates stay synthetic
    passages whatever the questions, so a file changes what is asked, not what
    is ranked.

    Args:
        question_count: Distinct synthetic questions to draw from; ignored with
            a ``questions_file``.
        candidates: Synthetic passages each rerank request carries, as the
            workflow retrieves (``RAG_TOP_K_RETRIEVE``).
        top_n: The ``top_n`` each rerank request asks for (``RAG_TOP_K_RERANK``).
        questions_file: A file read by ``load_questions``; ``None`` for the
            synthetic questions.

    Returns:
        The pool, its questions and passages as tuples.

    Example:
        >>> build_request_pool(50, settings.top_k_retrieve, settings.top_k_rerank,
        ...                    questions_file=Path("data/quebec/questions.jsonl"))
    """
    passages = [p.text for d in synthetic_corpus(candidates) for p in chunk(d)]
    questions = (
        load_questions(questions_file)
        if questions_file
        else tuple(synthetic_questions(question_count))
    )
    return RequestPool(
        questions=questions,
        rerank_documents=tuple(passages[:candidates]),
        top_n=top_n,
    )


def load_questions(path: Path) -> tuple[str, ...]:
    """Read a questions file: JSON lines with a ``question`` key, or plain text.

    A file whose first non-blank line starts with ``{`` is JSON lines (what
    ``rag-fetch-corpus`` writes); any other is one question per line. Blank lines
    are skipped and surrounding spaces stripped.

    Args:
        path: The file, from ``RAG_LOADTEST_QUESTIONS_FILE``.

    Returns:
        The questions, in file order.

    Raises:
        ValueError: If the file cannot be read or holds no question, or a JSON
            line is not an object with a non-empty string ``question``; the
            message names the file, and the line.

    Example:
        >>> load_questions(Path("data/quebec/questions.jsonl"))[0]
        'Qui paie ...?'
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{QUESTIONS_FILE_VAR}={str(path)!r}: {exc}") from exc
    lines = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1)]
    lines = [(n, line) for n, line in lines if line]
    if not lines:
        raise ValueError(f"{QUESTIONS_FILE_VAR}={str(path)!r}: no question in it")
    if not lines[0][1].startswith("{"):
        return tuple(line for _, line in lines)
    return tuple(_json_question(path, n, line) for n, line in lines)


def _json_question(path: Path, lineno: int, line: str) -> str:
    # The non-empty string ``question`` of one JSON line, or an error naming it.
    try:
        question = json.loads(line).get("question")
    except (ValueError, AttributeError):  # not JSON, or not an object
        question = None
    if not isinstance(question, str) or not question.strip():
        detail = 'expected a JSON object with a non-empty "question"'
        raise ValueError(f"{path}: line {lineno}: {detail}, got {line[:80]!r}")
    return question.strip()


@dataclass(frozen=True)
class RequestPlan:
    """One request to send: where, with which JSON body, under which stats name."""

    path: str
    payload: dict[str, object]
    stats_name: str


def plan_rag_request(
    endpoint: str, pool: RequestPool, rng: random.Random
) -> RequestPlan:
    """Plan a workflow-app request: one random question to ``/query`` or ``/retrieve``.

    Args:
        endpoint: The route, ``/query`` or ``/retrieve``; also the stats name.
        pool: Where the question is drawn from.
        rng: Picks the question.

    Returns:
        The plan, its payload ``{"question": ...}``.

    Example:
        >>> plan_rag_request("/query", pool, random.Random(7)).payload
        {'question': ...}
    """
    return RequestPlan(endpoint, {"question": rng.choice(pool.questions)}, endpoint)


def plan_rerank_request(
    alias: str, pool: RequestPool, rng: random.Random
) -> RequestPlan:
    """Plan a model-server rerank request shaped like the workflow's rerank stage.

    Args:
        alias: The model name the server serves the reranker under.
        pool: Where the query is drawn from, with the candidates and ``top_n``.
        rng: Picks the query.

    Returns:
        The plan for ``MODEL_SERVER_RERANK_PATH``, also its stats name.

    Example:
        >>> plan_rerank_request("reranker-alias", pool, random.Random(7)).path
    """
    request = CohereRerankRequest(
        model=alias,
        query=rng.choice(pool.questions),
        documents=list(pool.rerank_documents),
        top_n=pool.top_n,
    )
    path = MODEL_SERVER_RERANK_PATH
    return RequestPlan(path, request.to_wire(), path)


def plan_embeddings_request(
    alias: str, pool: RequestPool, rng: random.Random
) -> RequestPlan:
    """Plan a model-server embeddings request shaped like the workflow's embed stage.

    Args:
        alias: The model name the server serves the embedder under.
        pool: Where the one question to embed is drawn from.
        rng: Picks the question.

    Returns:
        The plan for ``MODEL_SERVER_EMBEDDINGS_PATH``, also its stats name.

    Example:
        >>> plan_embeddings_request("embedder-alias", pool, random.Random(7)).path
    """
    question = rng.choice(pool.questions)
    request = OpenAIEmbeddingsRequest(model=alias, input=[question])
    path = MODEL_SERVER_EMBEDDINGS_PATH
    return RequestPlan(path, request.to_wire(), path)
