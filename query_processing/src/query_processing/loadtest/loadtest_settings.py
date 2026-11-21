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
A request carries the API key of the key source its ``key`` block names
(``QP_LOADTEST_KEY__SOURCE``, ``QP_LOADTEST_KEY__VARIABLE``), the block an
endpoint of the workflow app writes; without one, ``DOMINO_API_KEY``, Domino's,
in its own header. The ``domino`` source asks the token URL and holds a token
for the time the workflow app reads (``QP_DOMINO__TOKEN_URL``,
``QP_DOMINO__TOKEN_TTL_S``), with its defaults. This is the one module of the
load-test client that imports the config package: for the defaults of the two
aliases, which it reads from the workflow file, and for the ``key`` and
``domino`` blocks; it never builds the RAG system's settings. A model-server
target posts to the /v3 path of OVMS and of our model server unless
``QP_LOADTEST_MODEL_SERVER_PATH`` names another, such as vLLM's
``/v1/rerank``: a path, not a model-server kind, since the kinds are the deploy
tool's, which this package does not import.

Example:
    >>> LoadTestSettings().target  # "rag" unless QP_LOADTEST_TARGET says otherwise
    'rag'
    >>> LoadTestSettings(key={"source": "domino"}).key.source
    'domino'
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ..config.components.key_blocks import KeyBlocksChecked, check_key_block
from ..config.components.key_settings import KeySettings
from ..config.files.workflow_file import declared_alias
from ..config.shared_settings import DefaultHttpSettings, DominoSettings

LoadTarget = Literal["rag", "rerank", "embeddings"]
LOAD_TARGETS: tuple[LoadTarget, ...] = ("rag", "rerank", "embeddings")
ENV_PREFIX = "QP_LOADTEST_"
# The shared http block's fields, whose defaults the recorder's retries reuse.
_HTTP_DEFAULTS = DefaultHttpSettings.model_fields
# Read before key sources; now refused unless a key block is set beside it.
_OLD_BEARER_TOKEN = "QP_LOADTEST_BEARER_TOKEN"
# The target that sends each alias a model server serves its model under.
_TARGET_OF = {"reranker": "rerank", "embedder": "embeddings"}


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
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- what to drive, and how hard ------------------------------------------------
    target: LoadTarget = "rag"  # rag | rerank | embeddings
    # The path a model-server target posts to; "" = its /v3 path (OVMS's and
    # qp-model-server's). A vLLM App answers on /v1/rerank and /v1/embeddings.
    model_server_path: str = ""
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
    # The largest difference between two scores of a vector-store record that
    # still counts as equal; a model run on other CPUs or threads moves the last
    # digits.
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
    # --- the API key a request carries: a key block wins over DOMINO_API_KEY ---------
    # Where the key comes from, as an endpoint's key block says it; None = none.
    key: KeySettings | None = None
    # The domino source's token URL and lifetime: the workflow app's variables.
    domino: DominoSettings = Field(
        default_factory=DominoSettings,
        validation_alias=_variable("QP_DOMINO", "domino"),
    )
    domino_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=_variable("DOMINO_API_KEY", "domino_api_key"),
    )
    # Read only to be refused without a key block; never sent.
    old_bearer_token: str = Field(
        default="",
        repr=False,
        validation_alias=_variable(_OLD_BEARER_TOKEN, "old_bearer_token"),
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

    @field_validator("target", mode="before")
    @classmethod
    def _known_target(cls, value: object) -> object:
        if value in LOAD_TARGETS:
            return value
        expected = ", ".join(LOAD_TARGETS)
        raise ValueError(f"QP_LOADTEST_TARGET={value!r}; expected one of {expected}")

    @field_validator("model_server_path", mode="before")
    @classmethod
    def _a_path_or_empty(cls, value: object) -> object:
        if value == "" or (isinstance(value, str) and value.startswith("/")):
            return value
        raise ValueError(
            f"QP_LOADTEST_MODEL_SERVER_PATH={value!r}; expected a path that starts "
            "with /, such as /v1/rerank, or empty for the target's /v3 path"
        )

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

    # The checks of two fields together are a field's, the later one's, never
    # the model's: a model's error quotes every value given, keys among them.
    @field_validator("wait_max_s")
    @classmethod
    def _no_shorter_than_the_least(cls, value: float, info: ValidationInfo) -> float:
        least = info.data.get("wait_min_s")
        if least is not None and value < least:
            raise ValueError(
                f"QP_LOADTEST_WAIT_MAX_S={value} is below "
                f"QP_LOADTEST_WAIT_MIN_S={least}; expected max >= min"
            )
        return value

    @field_validator("top_k_rerank")
    @classmethod
    def _rerank_sizes_that_run(
        cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        target = info.data.get("target")
        if target is not None and "top_k_retrieve" in info.data:
            _check_rerank_sizes(target, info.data["top_k_retrieve"], value)
        return value

    @field_validator("reranker_alias", "embedder_alias")
    @classmethod
    def _alias_from_the_workflow_file(cls, value: str, info: ValidationInfo) -> str:
        # Only a model-server target sends an alias, so only it reads the
        # workflow file: the workflow app's target needs none.
        component = str(info.field_name).removesuffix("_alias")
        sends_it = info.data.get("target") == _TARGET_OF[component]
        return declared_alias(component) if sends_it and not value else value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read as pydantic-settings reads, then check the key settings as read.

        Args:
            settings_cls: The settings class being built.
            init_settings: Constructor arguments.
            env_settings: The environment.
            dotenv_settings: The ``.env`` file.
            file_secret_settings: The secrets directory, none configured.

        Returns:
            The sources, highest priority first, ``KeySettingsChecked`` last.
        """
        read = (init_settings, env_settings, dotenv_settings, file_secret_settings)
        return (*read, KeySettingsChecked(settings_cls, cls.variable_of))


class KeySettingsChecked(KeyBlocksChecked):
    """The last settings source: refuses key mistakes in what the others gave.

    The workflow settings' check (``KeyBlocksChecked``) on the load test's own
    key settings. The settings library quotes the value of whatever it
    refuses, and a key typed in the wrong place would be printed. So the
    ``key`` block is checked as read (``check_key_block``), and
    ``QP_LOADTEST_BEARER_TOKEN`` without a block is refused, before any value
    is validated: each refusal is one line that quotes no value.

    Example:
        >>> LoadTestSettings(key="sk-1")
        ValueError: QP_LOADTEST_KEY (key) is not a key block: ...
    """

    def __call__(self) -> dict[str, Any]:
        """Check the key settings the sources before it gave; give nothing.

        Returns:
            An empty mapping.

        Raises:
            ValueError: A ``key`` block not shaped as one, or a bearer token
                without a block, in one line that quotes no value.
        """
        given = self.current_state
        block = given.get("key")
        check_key_block(block, variable=self._variable_of("key"), place="key")
        # The environment and .env give it by its variable, a caller by its field.
        token = given.get(_OLD_BEARER_TOKEN) or given.get("old_bearer_token")
        _refuse_a_bearer_token_alone(bool(token), block)
        return {}


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


def _refuse_a_bearer_token_alone(token: bool, block: object) -> None:
    # One line naming the block that replaces it, and never the token.
    if token and block is None:
        raise ValueError(
            f"{_OLD_BEARER_TOKEN} is no longer read: name the load test's key "
            "source in its key block instead, QP_LOADTEST_KEY__SOURCE=domino for "
            "the run token, asked for again as it expires, or "
            "QP_LOADTEST_KEY__SOURCE=env with QP_LOADTEST_KEY__VARIABLE=<the "
            "variable that holds the key>"
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
