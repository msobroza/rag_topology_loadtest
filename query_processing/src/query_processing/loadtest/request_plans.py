"""What a load-test user sends: the request pool, and one plan per request.

The questions come from ``QP_LOADTEST_QUESTIONS_FILE`` when it is set (what
``qp-fetch-corpus`` writes, or plain text), else they are synthetic; each is a
``/query`` body, which can carry a conversational history and a metadata filter
beside its question. A request to a model server carries what the workflow
app's stage would send, to the /v3 path unless ``QP_LOADTEST_MODEL_SERVER_PATH``
names another (vLLM's /v1 ones), and is counted under the /v3 path's name
whatever the path: ``qp-compare`` lines rows up by name, so a run on vLLM lines
up with the same run on OVMS. Values in, plans out: ``locustfile.py`` only does
the Locust I/O around them, so they are tested with plain calls and without
gevent monkey-patching the interpreter.

Example:
    >>> pool = build_request_pool(settings.question_count, *settings.rerank_sizes,
    ...                           questions_file=settings.questions_file)
    >>> plan = plan_rag_request("/query", pool, rng)
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from query_processing_core.query_request import QueryRequest

from ..domain.endpoints.api_endpoint import (
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_RERANK_PATH,
)
from ..domain.endpoints.standard_formats import (
    CohereRerankRequest,
    OpenAIEmbeddingsRequest,
)
from ..domain.retrieval.corpus import chunk, synthetic_corpus, synthetic_questions

QUESTIONS_FILE_VAR = "QP_LOADTEST_QUESTIONS_FILE"  # named in the reader's errors
_BODY_SHAPE = (
    'expected the body of a /query request: a JSON object with a "question",'
    ' and a "conversational_history" and a "metadata_filter" in the shape of the body'
)

# A line keeps these keys and leaves out any other (a corpus's evaluation
# material, say). Each body is checked against the app's own model of it, so
# a malformed one stops the load test at start-up naming its line rather than
# as 422s during the run; the conditions against the store's fields stay the
# app's to check, since only it knows them.
_BODY_KEYS = tuple(QueryRequest.model_fields)


@dataclass(frozen=True)
class RequestPool:
    """What requests draw from: the questions, the rerank candidates and ``top_n``.

    Each question is a ``/query`` body: its ``question``, with the
    ``conversational_history`` and ``metadata_filter`` a line of the questions
    file gave it. The candidates and ``top_n`` are what the workflow's rerank
    stage would send (``QP_TOP_K_RETRIEVE`` / ``_RERANK``).
    """

    questions: Sequence[dict[str, object]]
    rerank_documents: Sequence[str]
    top_n: int


def build_request_pool(
    question_count: int,
    candidates: int,
    top_n: int,
    *,
    questions_file: Path | None = None,
) -> RequestPool:
    """Build the pool: the questions, and the first ``candidates`` synthetic texts.

    The questions are every one of ``questions_file`` when it is given, else
    ``question_count`` synthetic ones, each a body with its question alone. The
    rerank and embeddings targets draw their queries from the same questions;
    the rerank candidates stay synthetic texts whatever the questions, so a
    file changes what is asked, not what is ranked.

    Args:
        question_count: Distinct synthetic questions to draw from; ignored with
            a ``questions_file``.
        candidates: Synthetic texts each rerank request carries, as the
            workflow retrieves (``QP_TOP_K_RETRIEVE``).
        top_n: The ``top_n`` each rerank request asks for (``QP_TOP_K_RERANK``).
        questions_file: A file read by ``load_questions``; ``None`` for the
            synthetic questions.

    Returns:
        The pool, its questions and texts as tuples.

    Example:
        >>> build_request_pool(50, *settings.rerank_sizes,
        ...                    questions_file=Path("data/quebec/questions.jsonl"))
    """
    texts = [r.text for d in synthetic_corpus(candidates) for r in chunk(d)]
    questions: tuple[dict[str, object], ...] = (
        load_questions(questions_file)
        if questions_file
        else tuple({"question": q} for q in synthetic_questions(question_count))
    )
    return RequestPool(
        questions=questions,
        rerank_documents=tuple(texts[:candidates]),
        top_n=top_n,
    )


def load_questions(path: Path) -> tuple[dict[str, object], ...]:
    """Read a questions file into ``/query`` bodies: JSON lines, or plain text.

    A file whose first non-blank line starts with ``{`` is JSON lines (what
    ``qp-fetch-corpus`` writes): each an object with a ``question`` and,
    optionally, the ``conversational_history`` and ``metadata_filter`` the
    request sends with it, in the shape the body takes; any other key is left
    out. The file thus decides the mix of first messages and follow-ups. Any
    other file is one question per line, each a body with its question alone.
    Blank lines are skipped and surrounding spaces stripped.

    Args:
        path: The file, from ``QP_LOADTEST_QUESTIONS_FILE``.

    Returns:
        The bodies, in file order.

    Raises:
        ValueError: If the file cannot be read or holds no question, or a
            line is not a body the workflow app takes (``QueryRequest``): a
            JSON line not an object with a ``question`` and a history and
            filter of the body's shape, or a question empty or too long; the
            message names the file, and the line.

    Example:
        >>> load_questions(Path("data/quebec/questions.jsonl"))[0]
        {'question': 'Qui paie ...?'}
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{QUESTIONS_FILE_VAR}={str(path)!r}: {exc}") from exc
    lines = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1)]
    lines = [(n, line) for n, line in lines if line]
    if not lines:
        raise ValueError(f"{QUESTIONS_FILE_VAR}={str(path)!r}: no question in it")
    body_of_line = _json_body if lines[0][1].startswith("{") else _text_body
    return tuple(_checked(path, n, line, body_of_line) for n, line in lines)


def _checked(
    path: Path, lineno: int, line: str, body_of_line: Callable[[str], dict[str, object]]
) -> dict[str, object]:
    # The body of one line, checked as the app checks it, or an error naming
    # the line.
    try:
        body = body_of_line(line)
        QueryRequest.model_validate(body)
    except ValueError as exc:  # not JSON, or not a body (ValidationError is one)
        got = f"got {line[:80]!r}{_failed_field(exc)}"
        raise ValueError(f"{path}: line {lineno}: {_BODY_SHAPE}, {got}") from exc
    return body


def _text_body(line: str) -> dict[str, object]:
    # A plain-text line is a question alone.
    return {"question": line}


def _json_body(line: str) -> dict[str, object]:
    # The body of one JSON line, as written but for its other keys and the
    # spaces around its question.
    raw = json.loads(line)
    if not isinstance(raw, dict):
        raise ValueError(f"{type(raw).__name__} is no JSON object")
    body = {key: raw[key] for key in _BODY_KEYS if key in raw}
    question = body.get("question")
    if isinstance(question, str):
        body["question"] = question.strip()
    return body


def _failed_field(exc: ValueError) -> str:
    # Where a body that is not of its shape fails and why, e.g. " at
    # question: String should have at most ... characters"; nothing for a line
    # that is not JSON.
    if not isinstance(exc, ValidationError):
        return ""
    error = exc.errors()[0]
    return f" at {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"


def _drawn_question(pool: RequestPool, rng: random.Random) -> str:
    # The question of one body drawn from the pool: a model server's query.
    return str(rng.choice(pool.questions)["question"])


@dataclass(frozen=True)
class RequestPlan:
    """One request to send: where, with which JSON body, under which stats name."""

    path: str
    payload: dict[str, object]
    stats_name: str


def plan_rag_request(
    endpoint: str, pool: RequestPool, rng: random.Random
) -> RequestPlan:
    """Plan a workflow-app request: one random body to ``/query`` or ``/retrieve``.

    The body is sent as the pool holds it: its question, with the
    conversational history and the metadata filter its line gave it.

    Args:
        endpoint: The route, ``/query`` or ``/retrieve``; also the stats name.
        pool: Where the body is drawn from.
        rng: Picks the body.

    Returns:
        The plan, its payload a copy of the drawn body.

    Example:
        >>> plan_rag_request("/query", pool, random.Random(7)).payload
        {'question': ...}
    """
    return RequestPlan(endpoint, dict(rng.choice(pool.questions)), endpoint)


def plan_rerank_request(
    alias: str, pool: RequestPool, rng: random.Random, path: str = ""
) -> RequestPlan:
    """Plan a model-server rerank request shaped like the workflow's rerank stage.

    Args:
        alias: The model name the server serves the reranker under.
        pool: Where the query is drawn from, with the candidates and ``top_n``.
        rng: Picks the query.
        path: Where the server answers (``QP_LOADTEST_MODEL_SERVER_PATH``);
            ``""`` for ``MODEL_SERVER_RERANK_PATH``.

    Returns:
        The plan for that path, under the stats name ``MODEL_SERVER_RERANK_PATH``
        whatever the path.

    Example:
        >>> plan_rerank_request("reranker-alias", pool, random.Random(7),
        ...                     "/v1/rerank").path
        '/v1/rerank'
    """
    request = CohereRerankRequest(
        model=alias,
        query=_drawn_question(pool, rng),
        documents=list(pool.rerank_documents),
        top_n=pool.top_n,
    )
    row = MODEL_SERVER_RERANK_PATH
    return RequestPlan(path or row, request.to_wire(), row)


def plan_embeddings_request(
    alias: str, pool: RequestPool, rng: random.Random, path: str = ""
) -> RequestPlan:
    """Plan a model-server embeddings request shaped like the workflow's embed stage.

    Args:
        alias: The model name the server serves the embedder under.
        pool: Where the one question to embed is drawn from.
        rng: Picks the question.
        path: Where the server answers (``QP_LOADTEST_MODEL_SERVER_PATH``);
            ``""`` for ``MODEL_SERVER_EMBEDDINGS_PATH``.

    Returns:
        The plan for that path, under the stats name
        ``MODEL_SERVER_EMBEDDINGS_PATH`` whatever the path.

    Example:
        >>> plan_embeddings_request("embedder-alias", pool, random.Random(7)).path
        '/v3/embeddings'
    """
    request = OpenAIEmbeddingsRequest(model=alias, input=[_drawn_question(pool, rng)])
    row = MODEL_SERVER_EMBEDDINGS_PATH
    return RequestPlan(path or row, request.to_wire(), row)
