"""Locust users for ``make loadtest`` (RAG service) and ``make loadtest-models``.

A thin adapter: the run's configuration is ``LoadTestSettings``, and what to
send, what a response becomes in the statistics and the target record are
decided in ``query_processing.loadtest`` (``request_plans``, ``response_checks``,
``target_record``); this file does the I/O and reads nothing else.
``QP_LOADTEST_TARGET`` picks the user class: ``rag`` (default) drives the
workflow app's ``/query`` and ``/retrieve``; ``rerank`` and ``embeddings`` drive a
model server directly. Rerank requests carry ``QP_TOP_K_RETRIEVE`` documents and
``QP_TOP_K_RERANK`` as ``top_n``, like the workflow; ``QP_LOADTEST_QUESTIONS``
sets how many distinct synthetic questions the users ask, unless
``QP_LOADTEST_QUESTIONS_FILE`` names a file of questions to ask instead (what
``qp-fetch-corpus`` writes, or plain text). Each user waits
``QP_LOADTEST_WAIT_MIN_S`` to ``_MAX_S`` between requests, and waits for an
answer at most ``QP_LOADTEST_REQUEST_TIMEOUT_S``;
``QP_LOADTEST_SHAPE=step`` steps through ``QP_LOADTEST_STEP_USERS`` users,
``QP_LOADTEST_STEP_SECONDS`` each. As a run stops, its target record is written,
with the tokens of each stage of its responses, and its statistics file is
rewritten with the final numbers.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

from locust import FastHttpUser, LoadTestShape, between, events, task
from locust import stats as locust_stats

from query_processing.loadtest.loadtest_settings import LoadTestSettings
from query_processing.loadtest.request_plans import (
    RequestPlan,
    build_request_pool,
    plan_embeddings_request,
    plan_rag_request,
    plan_rerank_request,
)
from query_processing.loadtest.response_checks import (
    STAGE_REQUEST_TYPE,
    ResponseVerdict,
    judge_response,
)
from query_processing.loadtest.target_record import (
    StageTokenTotals,
    TargetPair,
    count_spent_tokens,
    target_pair,
    write_target_record,
)

SETTINGS = LoadTestSettings()
POOL = build_request_pool(
    SETTINGS.question_count,
    *SETTINGS.rerank_sizes,
    questions_file=SETTINGS.questions_file,
)
WAIT_TIME = between(SETTINGS.wait_min_s, SETTINGS.wait_max_s)
RNG = random.Random()
# Responses per (topology alias, run label) for the target record; one process,
# one run.
RESPONSES_SEEN: Counter[TargetPair] = Counter()
# The tokens of the run's responses by STAGE row name, for the target record;
# one process, one run.
TOKENS_SEEN: dict[str, StageTokenTotals] = {}


def send(user: FastHttpUser, plan: RequestPlan) -> None:
    """POST ``plan`` and record the response as ``judge_response`` decides.

    A response that names its target also counts towards the target record, and
    each stage that reports its tokens towards that stage's token totals.

    Args:
        user: The Locust user whose client sends and whose environment fires the
            STAGE rows.
        plan: The path, JSON body and stats name of the request.
    """
    with user.client.post(
        plan.path,
        json=plan.payload,
        headers=SETTINGS.auth_headers,
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
        record_success(user, verdict)


def record_success(user: FastHttpUser, verdict: ResponseVerdict) -> None:
    """Fire a success's STAGE rows and add its stages' tokens to the run's totals.

    Args:
        user: The Locust user whose environment fires the STAGE rows.
        verdict: What ``judge_response`` made of the successful response.
    """
    count_spent_tokens(TOKENS_SEEN, verdict.tokens)
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
    """POST /query and /retrieve with random questions, weighted by the settings."""

    abstract = SETTINGS.target != "rag"
    wait_time = WAIT_TIME
    network_timeout = SETTINGS.request_timeout_s

    @task(SETTINGS.query_weight)
    def query(self) -> None:
        """Send one random question to ``/query``."""
        send(self, plan_rag_request("/query", POOL, RNG))

    @task(SETTINGS.retrieve_weight)
    def retrieve(self) -> None:
        """Send one random question to ``/retrieve``."""
        send(self, plan_rag_request("/retrieve", POOL, RNG))


class RerankUser(FastHttpUser):
    """POST /v3/rerank with the workflow's candidate count and top_n."""

    abstract = SETTINGS.target != "rerank"
    wait_time = WAIT_TIME
    network_timeout = SETTINGS.request_timeout_s

    @task
    def rerank(self) -> None:
        """Send one rerank request for the served reranker alias."""
        send(self, plan_rerank_request(SETTINGS.reranker_alias, POOL, RNG))


class EmbeddingsUser(FastHttpUser):
    """POST /v3/embeddings with one question, like the workflow's embed stage."""

    abstract = SETTINGS.target != "embeddings"
    wait_time = WAIT_TIME
    network_timeout = SETTINGS.request_timeout_s

    @task
    def embed(self) -> None:
        """Send one embeddings request for the served embedder alias."""
        send(self, plan_embeddings_request(SETTINGS.embedder_alias, POOL, RNG))


if SETTINGS.step_shape:
    STEP_USERS, STEP_SECONDS = SETTINGS.step_users, SETTINGS.step_seconds

    class StepLoadShape(LoadTestShape):  # locust picks up shape classes defined here
        """Ramp through ``QP_LOADTEST_STEP_USERS``, one step per ``_STEP_SECONDS``."""

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
    write_target_record(csv_prefix, SETTINGS.target, RESPONSES_SEEN, TOKENS_SEEN)


@events.test_stop.add_listener
def write_final_stats_on_stop(environment: Any, **_kwargs: Any) -> None:
    """Rewrite ``<--csv prefix>_stats.csv`` with the finished run's statistics.

    Locust rewrites that file once a second while it runs and only closes it as
    it shuts down (``StatsCSVFileWriter`` in Locust 2.47). A run that ends between
    two writes, as one stopped by ``--run-time`` does, would leave its last
    partial second out of the file ``qp-compare`` reads. Every user has stopped
    when ``test_stop`` fires, so the statistics are final here. Nothing is
    written without ``--csv``.

    Args:
        environment: The Locust environment, for its statistics and its parsed
            ``--csv`` prefix.
        **_kwargs: The rest of the ``test_stop`` event's arguments, unused.
    """
    csv_prefix = getattr(environment.parsed_options, "csv_prefix", None)
    if not csv_prefix:
        return
    final = locust_stats.StatsCSV(environment, locust_stats.PERCENTILES_TO_REPORT)
    # Opened as Locust opens it and written by Locust's own writer: the same
    # columns and line endings, so if Locust writes once more before it exits,
    # it writes these same rows over these. Its handle holds nothing unwritten:
    # each of its writes ends with a truncate, which flushes.
    with Path(f"{csv_prefix}_stats.csv").open("w") as handle:
        final.requests_csv(csv.writer(handle))
