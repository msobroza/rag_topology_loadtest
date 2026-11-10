"""Locust users for ``make loadtest`` (RAG service) and ``make loadtest-models``.

A thin adapter: what to send, what a response becomes in the statistics and the
target record are decided in ``rag_load_test.loadtest.request_plans``; this file
does the I/O.
``RAG_LOADTEST_TARGET`` picks the user class: ``rag`` (default) drives the
workflow app's ``/query`` and ``/retrieve``; ``rerank`` and ``embeddings`` drive a
model server directly. ``RAG_LOADTEST_SHAPE=step`` enables the 5 -> 10 -> 20 -> 40
user ramp.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any

from locust import FastHttpUser, LoadTestShape, between, events, task

from rag_load_test.config.settings import RagSettings
from rag_load_test.loadtest.request_plans import (
    STAGE_REQUEST_TYPE,
    RequestPlan,
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
RNG = random.Random()
# Responses per (topology, run label) for the target record; one process, one run.
RESPONSES_SEEN: Counter[tuple[str, str]] = Counter()


def send(user: FastHttpUser, plan: RequestPlan) -> None:
    """POST ``plan`` and record the response as ``judge_response`` decides."""
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
    abstract = CONFIG.target != "rag"
    wait_time = between(0.2, 1.0)

    @task(CONFIG.query_weight)
    def query(self) -> None:
        send(self, plan_rag_request("/query", RNG))

    @task(CONFIG.retrieve_weight)
    def retrieve(self) -> None:
        send(self, plan_rag_request("/retrieve", RNG))


class RerankUser(FastHttpUser):
    """POST /v3/rerank with the workflow's candidate count and top_n."""

    abstract = CONFIG.target != "rerank"
    wait_time = between(0.2, 1.0)

    @task
    def rerank(self) -> None:
        send(self, plan_rerank_request(SETTINGS.ovms_rerank_model, RNG))


class EmbeddingsUser(FastHttpUser):
    """POST /v3/embeddings with one question, like the workflow's embed stage."""

    abstract = CONFIG.target != "embeddings"
    wait_time = between(0.2, 1.0)

    @task
    def embed(self) -> None:
        send(self, plan_embeddings_request(SETTINGS.ovms_embeddings_model, RNG))


if CONFIG.step_shape:
    STEP_USERS, STEP_SECONDS = (5, 10, 20, 40), 60

    class StepLoadShape(LoadTestShape):  # locust picks up shape classes defined here
        def tick(self) -> tuple[int, float] | None:
            step = int(self.get_run_time() // STEP_SECONDS)
            if step >= len(STEP_USERS):
                return None
            return STEP_USERS[step], float(STEP_USERS[step])


@events.test_stop.add_listener
def write_target_record_on_stop(environment: Any, **_kwargs: Any) -> None:
    """Write ``<--csv prefix>_target.json`` as the run stops; nothing without --csv."""
    csv_prefix = getattr(environment.parsed_options, "csv_prefix", None)
    write_target_record(csv_prefix, CONFIG.target, RESPONSES_SEEN)
