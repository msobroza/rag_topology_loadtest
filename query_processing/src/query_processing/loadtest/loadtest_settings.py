"""The load-test client's configuration: every variable a run reads, checked once.

``locustfile.py`` builds one ``LoadTestSettings()`` at import, from the
environment and ``.env``, as the RAG system's and the deploy tool's settings are
read. Most variables are ``QP_LOADTEST_*``. Four are what a request sent to a
model server directly carries, as the workflow app would send it, and only a
model-server target reads them: the aliases of the ``reranker`` and
``embedder`` components (``QP_COMPONENTS__RERANKER__ALIAS``,
``QP_COMPONENTS__EMBEDDER__ALIAS``), which default to the aliases the
workflow file gives them, and the documents and ``top_n`` of a rerank request
(``QP_TOP_K_RETRIEVE``, ``QP_TOP_K_RERANK``), which have no default: they
are the use case's sizes, which its environment example states. The workflow
app's own target reads none of them, and so needs no workflow file.
``DOMINO_API_KEY`` is Domino's. This is the one module of the load-test client
that imports the config package, and only for the defaults of the two aliases:
it reads them from the workflow file and never builds the RAG system's
settings.

Example:
    >>> LoadTestSettings().target  # "rag" unless QP_LOADTEST_TARGET says otherwise
    'rag'
    >>> LoadTestSettings(domino_api_key="key").auth_headers
    {'X-Domino-Api-Key': 'key'}
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ..config.shared_settings import DefaultHttpSettings
from ..config.workflow_file import declared_alias

LoadTarget = Literal["rag", "rerank", "embeddings"]
LOAD_TARGETS: tuple[LoadTarget, ...] = ("rag", "rerank", "embeddings")
ENV_PREFIX = "QP_LOADTEST_"
# The shared http block's fields, whose defaults the recorder's retries reuse.
_HTTP_DEFAULTS = DefaultHttpSettings.model_fields


def _variable(name: str, field: str) -> AliasChoices:
    # A variable outside QP_LOADTEST_*; the field name stays usable as a keyword.
    return AliasChoices(name, field)


class LoadTestSettings(BaseSettings):
    """Every variable a load-test run reads, one field each.

    Construction checks every value, so a bad one fails when Locust imports its
    file, with a message naming the variable, the value and what was expected.

    Example:
        >>> LoadTestSettings(shape="step").step_users
        (5, 10, 20, 40)
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=".env", extra="ignore"
    )

    # --- what to drive, and how hard ------------------------------------------------
    target: LoadTarget = "rag"  # rag | rerank | embeddings
    query_weight: int = 1  # Locust task weights of RagUser; 0 turns a task off
    retrieve_weight: int = 3
    wait_min_s: float = Field(default=0.2, ge=0)  # each user waits between requests
    wait_max_s: float = Field(default=1.0, ge=0)
    # How long a user waits for an answer before Locust records a failure; an
    # answer of a few hundred tokens on CPUs can take longer than a minute.
    request_timeout_s: float = Field(default=60.0, gt=0)
    # qp-record-answers retries a request the target never answered, waiting
    # backoff_s before the first retry and twice as long before each next one;
    # the defaults are the workflow file's shared http block's.
    retries: int = Field(default=_HTTP_DEFAULTS["retries"].default, ge=0)
    backoff_s: float = Field(default=_HTTP_DEFAULTS["backoff_s"].default, ge=0)
    # --- what qp-compare-answers holds equal -----------------------------------------
    # The largest difference between two scores of a passage that still counts
    # as equal; a model run on other CPUs or threads moves the last digits.
    score_tolerance: float = Field(default=1e-4, ge=0)
    # An LLM can word an answer otherwise at temperature 0, so its text is
    # never compared exactly: with this on, an answer is the baseline's when its
    # ROUGE-L F-measure against it is at or above the threshold.
    answer_rouge_l: bool = False
    answer_rouge_l_threshold: float = 0.9
    # --- the load shape: unset = Locust's -u and -r; step = the steps below -------
    shape: Literal["step"] | None = None
    step_users: Annotated[tuple[int, ...], NoDecode] = (5, 10, 20, 40)  # "5,10,20,40"
    step_seconds: float = Field(default=60.0, gt=0)
    # --- what the users ask -----------------------------------------------------------
    question_count: int = Field(
        default=50,
        validation_alias=_variable("QP_LOADTEST_QUESTIONS", "question_count"),
    )
    questions_file: Path | None = None  # replaces the synthetic questions when set
    # --- what a model server is sent: the RAG system's variables ----------------------
    # The workflow app no longer reads these two: its sizes are settings of its
    # stages, in the workflow file. They are the use case's, so they have no
    # default here: the rerank target requires both (None = unset).
    top_k_retrieve: int | None = Field(
        default=None,
        ge=1,
        validation_alias=_variable("QP_TOP_K_RETRIEVE", "top_k_retrieve"),
    )
    top_k_rerank: int | None = Field(
        default=None,
        ge=1,
        validation_alias=_variable("QP_TOP_K_RERANK", "top_k_rerank"),
    )
    # The names a model server serves its model under; "" = the workflow
    # file's, read when the settings are built for the target that sends it.
    reranker_alias: str = Field(
        default="",
        validation_alias=_variable("QP_COMPONENTS__RERANKER__ALIAS", "reranker_alias"),
    )
    embedder_alias: str = Field(
        default="",
        validation_alias=_variable("QP_COMPONENTS__EMBEDDER__ALIAS", "embedder_alias"),
    )
    # --- Domino auth: a bearer token wins over an API key ----------------------------
    bearer_token: str = ""
    domino_api_key: str = Field(
        default="", validation_alias=_variable("DOMINO_API_KEY", "domino_api_key")
    )

    @classmethod
    def variable_of(cls, field: str) -> str:
        """Return the environment variable a field is read from.

        Args:
            field: The field name.

        Returns:
            Its own variable when it has one, else ``QP_LOADTEST_<FIELD>``.

        Example:
            >>> LoadTestSettings.variable_of("question_count")
            'QP_LOADTEST_QUESTIONS'
        """
        alias = cls.model_fields[field].validation_alias
        if isinstance(alias, AliasChoices):
            return str(alias.choices[0])
        return ENV_PREFIX + field.upper()

    @property
    def step_shape(self) -> bool:
        """Whether the run follows the step shape (``QP_LOADTEST_SHAPE=step``)."""
        return self.shape == "step"

    @property
    def rerank_sizes(self) -> tuple[int, int]:
        """The documents and ``top_n`` of a rerank request, 0 each where unset.

        Only the rerank target sends one, and it requires both sizes.
        """
        return self.top_k_retrieve or 0, self.top_k_rerank or 0

    @property
    def auth_headers(self) -> dict[str, str]:
        """The Domino auth header: a bearer token wins over an API key, else none."""
        if self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        if self.domino_api_key:
            return {"X-Domino-Api-Key": self.domino_api_key}
        return {}

    @field_validator("target", mode="before")
    @classmethod
    def _known_target(cls, value: object) -> object:
        if value in LOAD_TARGETS:
            return value
        expected = ", ".join(LOAD_TARGETS)
        raise ValueError(f"QP_LOADTEST_TARGET={value!r}; expected one of {expected}")

    @field_validator("query_weight", "retrieve_weight", "question_count", mode="before")
    @classmethod
    def _whole_number(cls, value: object, info: ValidationInfo) -> int:
        minimum = 1 if info.field_name == "question_count" else 0
        return _at_least(cls.variable_of(str(info.field_name)), value, minimum)

    @field_validator("answer_rouge_l_threshold", mode="before")
    @classmethod
    def _between_0_and_1(cls, value: object) -> float:
        variable = cls.variable_of("answer_rouge_l_threshold")
        message = f"{variable}={value!r}; expected a number between 0 and 1"
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(message)
        try:
            number = float(value)
        except ValueError:
            raise ValueError(message) from None
        if not 0 <= number <= 1:
            raise ValueError(message)
        return number

    @field_validator("shape", mode="before")
    @classmethod
    def _step_or_unset(cls, value: object) -> object:
        if value in (None, "", "step"):
            return value or None
        raise ValueError(f"QP_LOADTEST_SHAPE={value!r}; expected 'step' or unset")

    @field_validator("step_users", mode="before")
    @classmethod
    def _user_steps(cls, value: object) -> tuple[int, ...]:
        steps = value.split(",") if isinstance(value, str) else value
        if not isinstance(steps, Sequence) or not steps:
            raise ValueError(_steps_message(value))
        try:
            return tuple(_at_least("QP_LOADTEST_STEP_USERS", s, 1) for s in steps)
        except ValueError:
            raise ValueError(_steps_message(value)) from None

    @field_validator("questions_file", mode="before")
    @classmethod
    def _unset_when_empty(cls, value: object) -> object:
        return value or None

    @model_validator(mode="after")
    def _consistent(self) -> LoadTestSettings:
        if self.wait_max_s < self.wait_min_s:
            raise ValueError(
                f"QP_LOADTEST_WAIT_MAX_S={self.wait_max_s} is below "
                f"QP_LOADTEST_WAIT_MIN_S={self.wait_min_s}; expected max >= min"
            )
        _check_rerank_sizes(self.target, self.top_k_retrieve, self.top_k_rerank)
        return self

    @model_validator(mode="after")
    def _aliases_from_the_workflow_file(self) -> LoadTestSettings:
        # Only a model-server target sends an alias, so only it reads the
        # workflow file: the workflow app's target needs none.
        if self.target == "rerank" and not self.reranker_alias:
            self.reranker_alias = declared_alias("reranker")
        if self.target == "embeddings" and not self.embedder_alias:
            self.embedder_alias = declared_alias("embedder")
        return self


def _check_rerank_sizes(
    target: LoadTarget, candidates: int | None, top_n: int | None
) -> None:
    # The sizes of a rerank request are the use case's, so they have no default.
    if target == "rerank" and (candidates is None or top_n is None):
        raise ValueError(
            "QP_LOADTEST_TARGET=rerank needs QP_TOP_K_RETRIEVE and "
            "QP_TOP_K_RERANK, the documents and top_n of each rerank request, "
            "as the workflow file's retrieve and rerank stages set them (top_k); "
            f"got {candidates} and {top_n}"
        )
    # A request with a top_n above its candidates is one the workflow app
    # never sends.
    if candidates is not None and top_n is not None and top_n > candidates:
        raise ValueError(
            f"QP_TOP_K_RERANK={top_n} exceeds QP_TOP_K_RETRIEVE={candidates}; "
            "expected at most it"
        )


def _at_least(variable: str, raw: object, minimum: int) -> int:
    # A whole number of at least ``minimum``, or an error naming the variable.
    message = f"{variable}={raw!r}; expected a whole number of {minimum} or more"
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ValueError(message)
    try:
        number = int(raw)
    except ValueError:
        raise ValueError(message) from None
    if number < minimum:
        raise ValueError(message)
    return number


def _steps_message(raw: object) -> str:
    return (
        f"QP_LOADTEST_STEP_USERS={raw!r}; expected comma-separated whole numbers "
        "of 1 or more, e.g. 5,10,20,40"
    )
