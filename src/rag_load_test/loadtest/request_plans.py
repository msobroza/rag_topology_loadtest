"""Client side of the load test: every Locust user decision, and ``rag-compare``.

Values in, decisions out: the run's configuration, what each user sends, what a
response becomes in the statistics (a failure, or STAGE rows), and the target
record. ``locustfile.py`` only does Locust I/O around these, so they are tested
with plain calls and without gevent monkey-patching the interpreter.

Example:
    >>> pool = build_request_pool(config.question_count, settings.top_k_retrieve,
    ...                           settings.top_k_rerank)
    >>> plan = plan_rag_request("/query", pool, rng)
    >>> verdict = judge_response(plan.path, resp.status_code, resp.text or "")
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

from ..adapters.ovms.v3_contract import (
    EMBEDDINGS_PATH,
    RERANK_PATH,
    EmbeddingsRequest,
    RerankRequest,
)
from ..domain.corpus import chunk, synthetic_corpus, synthetic_questions
from ..domain.passages import STAGES

# --- configuration: the environment, read once ---------------------------------

LOAD_TARGETS: tuple[str, ...] = ("rag", "rerank", "embeddings")


@dataclass(frozen=True)
class LoadTestConfig:
    """Everything a run reads from its environment, validated up front."""

    target: str
    query_weight: int
    retrieve_weight: int
    step_shape: bool
    auth_headers: Mapping[str, str]
    question_count: int  # distinct synthetic questions the users pick from


def read_loadtest_config(env: Mapping[str, str]) -> LoadTestConfig:
    """Read and validate the ``RAG_LOADTEST_*`` knobs (and Domino auth) once.

    Defaults: target ``rag``, query weight 1, retrieve weight 3, 50 questions;
    ``RAG_LOADTEST_SHAPE=step`` turns on the step shape. Domino auth is
    ``RAG_LOADTEST_BEARER_TOKEN`` as a bearer token, else ``DOMINO_API_KEY``.

    Args:
        env: The environment to read, usually ``os.environ``.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If ``RAG_LOADTEST_TARGET`` is not one of ``LOAD_TARGETS``, a
            weight or the question count is not an integer, or the question
            count is below 1.

    Example:
        >>> read_loadtest_config(os.environ).target  # "rag" unless overridden
        'rag'
    """
    target = env.get("RAG_LOADTEST_TARGET", "rag")
    if target not in LOAD_TARGETS:
        allowed = ", ".join(LOAD_TARGETS)
        raise ValueError(f"RAG_LOADTEST_TARGET={target!r}; expected one of {allowed}")
    return LoadTestConfig(
        target=target,
        query_weight=_task_weight(env, "RAG_LOADTEST_QUERY_WEIGHT", 1),
        retrieve_weight=_task_weight(env, "RAG_LOADTEST_RETRIEVE_WEIGHT", 3),
        step_shape=env.get("RAG_LOADTEST_SHAPE") == "step",
        auth_headers=_auth_headers(env),
        question_count=_question_count(env),
    )


def _task_weight(env: Mapping[str, str], variable: str, default: int) -> int:
    raw = env.get(variable)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{variable}={raw!r}; expected an integer") from None


def _question_count(env: Mapping[str, str]) -> int:
    count = _task_weight(env, "RAG_LOADTEST_QUESTIONS", 50)
    if count < 1:
        raw = env["RAG_LOADTEST_QUESTIONS"]
        raise ValueError(f"RAG_LOADTEST_QUESTIONS={raw!r}; expected at least 1")
    return count


# Domino auth for callers outside a run: a bearer token wins over an API key.
def _auth_headers(env: Mapping[str, str]) -> dict[str, str]:
    if env.get("RAG_LOADTEST_BEARER_TOKEN"):
        return {"Authorization": f"Bearer {env['RAG_LOADTEST_BEARER_TOKEN']}"}
    if env.get("DOMINO_API_KEY"):
        return {"X-Domino-Api-Key": env["DOMINO_API_KEY"]}
    return {}


# --- what to send ---------------------------------------------------------------


@dataclass(frozen=True)
class RequestPool:
    """What requests draw from: the questions, the rerank candidates and ``top_n``.

    The candidates and ``top_n`` are what the workflow's rerank stage would send
    (``RAG_TOP_K_RETRIEVE`` / ``_RERANK``).
    """

    questions: Sequence[str]
    rerank_documents: Sequence[str]
    top_n: int


def build_request_pool(question_count: int, candidates: int, top_n: int) -> RequestPool:
    """Build the pool: synthetic questions, and the first ``candidates`` passages.

    Args:
        question_count: Distinct synthetic questions to draw from.
        candidates: Synthetic passages each rerank request carries, as the
            workflow retrieves (``RAG_TOP_K_RETRIEVE``).
        top_n: The ``top_n`` each rerank request asks for (``RAG_TOP_K_RERANK``).

    Returns:
        The pool, its questions and passages as tuples.

    Example:
        >>> build_request_pool(50, settings.top_k_retrieve, settings.top_k_rerank)
    """
    passages = [p.text for d in synthetic_corpus(candidates) for p in chunk(d)]
    return RequestPool(
        questions=tuple(synthetic_questions(question_count)),
        rerank_documents=tuple(passages[:candidates]),
        top_n=top_n,
    )


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
        The plan for ``RERANK_PATH``, also its stats name.

    Example:
        >>> plan_rerank_request("bge-reranker-base", pool, random.Random(7)).path
    """
    request = RerankRequest(
        model=alias,
        query=rng.choice(pool.questions),
        documents=list(pool.rerank_documents),
        top_n=pool.top_n,
    )
    return RequestPlan(RERANK_PATH, request.to_wire(), RERANK_PATH)


def plan_embeddings_request(
    alias: str, pool: RequestPool, rng: random.Random
) -> RequestPlan:
    """Plan a model-server embeddings request shaped like the workflow's embed stage.

    Args:
        alias: The model name the server serves the embedder under.
        pool: Where the one question to embed is drawn from.
        rng: Picks the question.

    Returns:
        The plan for ``EMBEDDINGS_PATH``, also its stats name.

    Example:
        >>> plan_embeddings_request("bge-small-en-v1.5", pool, random.Random(7)).path
    """
    request = EmbeddingsRequest(model=alias, input=[rng.choice(pool.questions)])
    return RequestPlan(EMBEDDINGS_PATH, request.to_wire(), EMBEDDINGS_PATH)


# --- what to record -------------------------------------------------------------

# The request type of STAGE rows, used both when firing them and by rag-compare.
STAGE_REQUEST_TYPE = "STAGE"


@dataclass(frozen=True)
class ResponseVerdict:
    """What one response becomes in the statistics: a failure, or STAGE rows.

    The STAGE rows are ``(name, ms)`` pairs to fire.
    """

    failure: str | None
    stage_rows: tuple[tuple[str, float], ...] = ()


def judge_response(endpoint: str, status: int, body_text: str) -> ResponseVerdict:
    """Decide how a response is recorded; only a 200 with a JSON object succeeds.

    Args:
        endpoint: The path requested; it prefixes each STAGE row name.
        status: The HTTP status of the response.
        body_text: The response body; failure messages quote its first 200
            characters.

    Returns:
        A failure message, or no failure and one STAGE row per stage timed above
        zero, in ``STAGES`` order (``total`` excluded); none when ``timings_ms``
        is missing or malformed.

    Example:
        >>> judge_response("/query", 200, '{"timings_ms": {"embed": 4}}').stage_rows
        (('/query:embed', 4.0),)
    """
    if status != HTTPStatus.OK:
        return ResponseVerdict(failure=f"HTTP {status}: {body_text[:200]}")
    try:
        body = json.loads(body_text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        message = f"HTTP 200 but the body is not a JSON object: {body_text[:200]}"
        return ResponseVerdict(failure=message)
    return ResponseVerdict(failure=None, stage_rows=_stage_rows(endpoint, body))


# One row per stage timed above zero, in ``STAGES`` order (``total`` excluded);
# none when ``timings_ms`` is missing or malformed.
def _stage_rows(
    endpoint: str, body: Mapping[str, Any]
) -> tuple[tuple[str, float], ...]:
    try:
        timings_ms = {str(k): float(v) for k, v in body["timings_ms"].items()}
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    return tuple(
        (f"{endpoint}:{s}", timings_ms[s]) for s in STAGES if timings_ms.get(s, 0) > 0
    )


# --- the target record: what the target said about itself during a run -------


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
    csv_prefix: str | None, target: str, counts: Mapping[tuple[str, str], int]
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
        >>> counts = Counter({("monolith", "m"): 3})
        >>> write_target_record("results/monolith", "rag", counts)
    """
    if not csv_prefix:
        return
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    seen = [
        {"topology": topology, "run_label": label, "responses": n}
        for (topology, label), n in ordered
        if target == "rag"
    ]
    document = {"target": target, "seen": seen}
    Path(f"{csv_prefix}_target.json").write_text(json.dumps(document, indent=2))
