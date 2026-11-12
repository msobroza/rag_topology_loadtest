"""Locust users for ``make loadtest`` (RAG service) and ``make loadtest-models``.

A thin adapter: what to send, what a response becomes in the statistics and the
target record are decided in ``rag_load_test.loadtest.request_plans``; this file
does the I/O.
``RAG_LOADTEST_TARGET`` picks the user class: ``rag`` (default) drives the
workflow app's ``/query`` and ``/retrieve``; ``rerank`` and ``embeddings`` drive a
model server directly. Rerank requests carry ``RAG_TOP_K_RETRIEVE`` documents and
``RAG_TOP_K_RERANK`` as ``top_n``, like the workflow; ``RAG_LOADTEST_QUESTIONS``
sets how many distinct synthetic questions the users ask, unless
``RAG_LOADTEST_QUESTIONS_FILE`` names a file of questions to ask instead (what
``rag-fetch-corpus`` writes, or plain text). ``RAG_LOADTEST_SHAPE=step`` enables
the 5 -> 10 -> 20 -> 40 user ramp.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any

from locust import FastHttpUser, LoadTestShape, between, events, task

from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.loadtest.request_plans import (
    STAGE_REQUEST_TYPE,
    RequestPlan,
    build_request_pool,
    judge_response,
    plan_embeddings_request,
    plan_rag_request,
    plan_rerank_request,
    read_loadtest_config,
    target_pair,
    write_target_record,
)

CONFIG = read_loadtest_config(os.environ)
SETTINGS = RagSettings()
POOL = build_request_pool(
    CONFIG.question_count,
    SETTINGS.top_k_retrieve,
    SETTINGS.top_k_rerank,
    questions_file=CONFIG.questions_file,
)
RNG = random.Random()
# Responses per (topology, run label) for the target record; one process, one run.
RESPONSES_SEEN: Counter[tuple[str, str]] = Counter()


def send(user: FastHttpUser, plan: RequestPlan) -> None:
    """POST ``plan`` and record the response as ``judge_response`` decides.

    A response that names its target also counts towards the target record.

    Args:
        user: The Locust user whose client sends and whose environment fires the
            STAGE rows.
        plan: The path, JSON body and stats name of the request.
    """
    with user.client.post(
        plan.path,
        json=plan.payload,
        headers=dict(CONFIG.auth_headers),
        name=plan.stats_name,
        catch_response=True,
    ) as resp:
        pair = target_pair(resp.headers)
        if pair is not None:
            RESPONSES_SEEN[pair] += 1
        verdict = judge_response(plan.path, resp.status_code, resp.text or "")
        if verdict.failure is not None:
            resp.failure(verdict.failure)
            return
        resp.success()
        for name, ms in verdict.stage_rows:
            user.environment.events.request.fire(
                request_type=STAGE_REQUEST_TYPE,
                name=name,
                response_time=ms,
                response_length=0,
                exception=None,
                context={},
            )


class RagUser(FastHttpUser):
    """POST /query and /retrieve with random questions, weighted by the config."""

    abstract = CONFIG.target != "rag"
    wait_time = between(0.2, 1.0)

    @task(CONFIG.query_weight)
    def query(self) -> None:
        """Send one random question to ``/query``."""
        send(self, plan_rag_request("/query", POOL, RNG))

    @task(CONFIG.retrieve_weight)
    def retrieve(self) -> None:
        """Send one random question to ``/retrieve``."""
        send(self, plan_rag_request("/retrieve", POOL, RNG))


class RerankUser(FastHttpUser):
    """POST /v3/rerank with the workflow's candidate count and top_n."""

    abstract = CONFIG.target != "rerank"
    wait_time = between(0.2, 1.0)

    @task
    def rerank(self) -> None:
        """Send one rerank request for the served reranker alias."""
        send(self, plan_rerank_request(SETTINGS.rerank_api_model, POOL, RNG))


class EmbeddingsUser(FastHttpUser):
    """POST /v3/embeddings with one question, like the workflow's embed stage."""

    abstract = CONFIG.target != "embeddings"
    wait_time = between(0.2, 1.0)

    @task
    def embed(self) -> None:
        """Send one embeddings request for the served embedder alias."""
        send(self, plan_embeddings_request(SETTINGS.embedding_api_model, POOL, RNG))


if CONFIG.step_shape:
    STEP_USERS, STEP_SECONDS = (5, 10, 20, 40), 60

    class StepLoadShape(LoadTestShape):  # locust picks up shape classes defined here
        """Ramp through 5, 10, 20 then 40 users, one step every 60 seconds."""

        def tick(self) -> tuple[int, float] | None:
            """Return the step's users and spawn rate, or ``None`` after the last step.

            Returns:
                ``(users, spawn rate)`` for the current step, the spawn rate equal
                to the users, or ``None`` to stop the run.
            """
            step = int(self.get_run_time() // STEP_SECONDS)
            if step >= len(STEP_USERS):
                return None
            return STEP_USERS[step], float(STEP_USERS[step])


@events.test_stop.add_listener
def write_target_record_on_stop(environment: Any, **_kwargs: Any) -> None:
    """Write ``<--csv prefix>_target.json`` as the run stops; nothing without --csv.

    Args:
        environment: The Locust environment, for its parsed ``--csv`` prefix.
        **_kwargs: The rest of the ``test_stop`` event's arguments, unused.
    """
    csv_prefix = getattr(environment.parsed_options, "csv_prefix", None)
    write_target_record(csv_prefix, CONFIG.target, RESPONSES_SEEN)
