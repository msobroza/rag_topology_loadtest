"""Locust users for ``make loadtest`` (RAG service) and ``make loadtest-models``.

``RAG_LOADTEST_TARGET`` picks the user class: ``rag`` (default) drives the
workflow app's ``/query`` and ``/retrieve``; ``rerank`` and ``embeddings`` drive a
model server (FastAPI or OVMS, same contract) directly on ``/v3/rerank`` and
``/v3/embeddings``. Each successful RAG response's ``timings_ms`` is re-emitted
as ``STAGE`` request events so per-stage p50/p95 land in the same ``*_stats.csv``.
``RAG_LOADTEST_SHAPE=step`` enables the 5 -> 10 -> 20 -> 40 user ramp.
"""

from __future__ import annotations

import os
import random

from locust import FastHttpUser, LoadTestShape, between, task

from rag_load_test.corpus import chunk, synthetic_corpus
from rag_load_test.loadtest import (
    QUESTIONS,
    auth_headers,
    parse_timings,
    stage_events,
    task_weights,
)
from rag_load_test.settings import RagSettings
from rag_load_test.workflow import TOP_K_RERANK, TOP_K_RETRIEVE

TARGET = os.environ.get("RAG_LOADTEST_TARGET", "rag")
QUERY_WEIGHT, RETRIEVE_WEIGHT = task_weights(os.environ)
SETTINGS = RagSettings()
# The same candidate count the workflow sends to its reranker.
RERANK_DOCUMENTS = [p.text for d in synthetic_corpus(TOP_K_RETRIEVE) for p in chunk(d)]


class RagUser(FastHttpUser):
    abstract = TARGET != "rag"
    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.headers = auth_headers(os.environ)

    @task(QUERY_WEIGHT)
    def query(self) -> None:
        self._ask("/query")

    @task(RETRIEVE_WEIGHT)
    def retrieve(self) -> None:
        self._ask("/retrieve")

    def _ask(self, path: str) -> None:
        payload = {"question": random.choice(QUESTIONS)}
        with self.client.post(
            path, json=payload, headers=self.headers, name=path, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {(resp.text or '')[:200]}")
                return
            for name, ms in stage_events(path, parse_timings(resp.json())):
                self.environment.events.request.fire(
                    request_type="STAGE",
                    name=name,
                    response_time=ms,
                    response_length=0,
                    exception=None,
                    context={},
                )


class RerankUser(FastHttpUser):
    """POST /v3/rerank with the workflow's candidate count and top_n."""

    abstract = TARGET != "rerank"
    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.headers = auth_headers(os.environ)

    @task
    def rerank(self) -> None:
        payload = {
            "model": SETTINGS.ovms_rerank_model,
            "query": random.choice(QUESTIONS),
            "documents": RERANK_DOCUMENTS[:TOP_K_RETRIEVE],
            "top_n": TOP_K_RERANK,
        }
        self.client.post(
            "/v3/rerank", json=payload, headers=self.headers, name="/v3/rerank"
        )


class EmbeddingsUser(FastHttpUser):
    """POST /v3/embeddings with one question, like the workflow's embed stage."""

    abstract = TARGET != "embeddings"
    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.headers = auth_headers(os.environ)

    @task
    def embed(self) -> None:
        payload = {
            "model": SETTINGS.ovms_embeddings_model,
            "input": [random.choice(QUESTIONS)],
        }
        self.client.post(
            "/v3/embeddings", json=payload, headers=self.headers, name="/v3/embeddings"
        )


if os.environ.get("RAG_LOADTEST_SHAPE") == "step":
    STEP_USERS, STEP_SECONDS = (5, 10, 20, 40), 60

    class StepLoadShape(LoadTestShape):  # locust picks up shape classes defined here
        def tick(self) -> tuple[int, float] | None:
            step = int(self.get_run_time() // STEP_SECONDS)
            if step >= len(STEP_USERS):
                return None
            return STEP_USERS[step], float(STEP_USERS[step])
