"""The load-test client's configuration: every variable a run reads, checked once.

``locustfile.py`` builds one ``LoadTestSettings()`` at import, from the
environment and ``.env``, as the RAG system's and the deploy tool's settings are
read. Most variables are ``RAG_LOADTEST_*``. Four are the RAG system's own
(``RAG_TOP_K_RETRIEVE``, ``RAG_TOP_K_RERANK``, ``RAG_RERANK_API_MODEL``,
``RAG_EMBEDDING_API_MODEL``): a request sent to a model server directly carries
what the workflow app would send, driven by the same variables, with the same
defaults. ``DOMINO_API_KEY`` is Domino's. This is the one module of the
load-test client that imports the config package, and only for those four
defaults: it never builds the RAG system's settings.

Example:
    >>> LoadTestSettings().target  # "rag" unless RAG_LOADTEST_TARGET says otherwise
    'rag'
    >>> LoadTestSettings(domino_api_key="key").auth_headers
    {'X-Domino-Api-Key': 'key'}
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ..config.rag_settings import RagSettings

LoadTarget = Literal["rag", "rerank", "embeddings"]
LOAD_TARGETS: tuple[LoadTarget, ...] = ("rag", "rerank", "embeddings")
ENV_PREFIX = "RAG_LOADTEST_"


def rag_default(field: str) -> Any:
    """Return the RAG system's default for one of its settings, without building them.

    Args:
        field: The name of a ``RagSettings`` field.

    Returns:
        The default that field declares.

    Example:
        >>> rag_default("top_k_retrieve")
        20
    """
    return RagSettings.model_fields[field].default


def _variable(name: str, field: str) -> AliasChoices:
    # A variable outside RAG_LOADTEST_*; the field name stays usable as a keyword.
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
    # --- the load shape: unset = Locust's -u and -r; step = the steps below -------
    shape: Literal["step"] | None = None
    step_users: Annotated[tuple[int, ...], NoDecode] = (5, 10, 20, 40)  # "5,10,20,40"
    step_seconds: float = Field(default=60.0, gt=0)
    # --- what the users ask -----------------------------------------------------------
    question_count: int = Field(
        default=50,
        validation_alias=_variable("RAG_LOADTEST_QUESTIONS", "question_count"),
    )
    questions_file: Path | None = None  # replaces the synthetic questions when set
    # --- what a model server is sent: the RAG system's variables and defaults ---------
    top_k_retrieve: int = Field(
        default=rag_default("top_k_retrieve"),
        ge=1,
        validation_alias=_variable("RAG_TOP_K_RETRIEVE", "top_k_retrieve"),
    )
    top_k_rerank: int = Field(
        default=rag_default("top_k_rerank"),
        ge=1,
        validation_alias=_variable("RAG_TOP_K_RERANK", "top_k_rerank"),
    )
    rerank_api_model: str = Field(
        default=rag_default("rerank_api_model"),
        validation_alias=_variable("RAG_RERANK_API_MODEL", "rerank_api_model"),
    )
    embedding_api_model: str = Field(
        default=rag_default("embedding_api_model"),
        validation_alias=_variable("RAG_EMBEDDING_API_MODEL", "embedding_api_model"),
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
            Its own variable when it has one, else ``RAG_LOADTEST_<FIELD>``.

        Example:
            >>> LoadTestSettings.variable_of("question_count")
            'RAG_LOADTEST_QUESTIONS'
        """
        alias = cls.model_fields[field].validation_alias
        if isinstance(alias, AliasChoices):
            return str(alias.choices[0])
        return ENV_PREFIX + field.upper()

    @property
    def step_shape(self) -> bool:
        """Whether the run follows the step shape (``RAG_LOADTEST_SHAPE=step``)."""
        return self.shape == "step"

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
        raise ValueError(f"RAG_LOADTEST_TARGET={value!r}; expected one of {expected}")

    @field_validator("query_weight", "retrieve_weight", "question_count", mode="before")
    @classmethod
    def _whole_number(cls, value: object, info: ValidationInfo) -> int:
        minimum = 1 if info.field_name == "question_count" else 0
        return _at_least(cls.variable_of(str(info.field_name)), value, minimum)

    @field_validator("shape", mode="before")
    @classmethod
    def _step_or_unset(cls, value: object) -> object:
        if value in (None, "", "step"):
            return value or None
        raise ValueError(f"RAG_LOADTEST_SHAPE={value!r}; expected 'step' or unset")

    @field_validator("step_users", mode="before")
    @classmethod
    def _user_steps(cls, value: object) -> tuple[int, ...]:
        steps = value.split(",") if isinstance(value, str) else value
        if not isinstance(steps, Sequence) or not steps:
            raise ValueError(_steps_message(value))
        try:
            return tuple(_at_least("RAG_LOADTEST_STEP_USERS", s, 1) for s in steps)
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
                f"RAG_LOADTEST_WAIT_MAX_S={self.wait_max_s} is below "
                f"RAG_LOADTEST_WAIT_MIN_S={self.wait_min_s}; expected max >= min"
            )
        # The RAG system's own rule: a request with a top_n above its candidates
        # is one the workflow app never sends.
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError(
                f"RAG_TOP_K_RERANK={self.top_k_rerank} exceeds "
                f"RAG_TOP_K_RETRIEVE={self.top_k_retrieve}; expected at most it"
            )
        return self


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
        f"RAG_LOADTEST_STEP_USERS={raw!r}; expected comma-separated whole numbers "
        "of 1 or more, e.g. 5,10,20,40"
    )
