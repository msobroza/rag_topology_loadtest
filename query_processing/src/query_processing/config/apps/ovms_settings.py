"""An OVMS model server App's own settings: streams, threads, start, CPU, batching.

OpenVINO Model Server serves one component per App, on that App's hardware, so
its stream and thread counts are the App's, not the workflow's: they are no
value of the workflow file. They are flat ``QP_OVMS_*`` variables, beside the
ones the launch script reads (``QP_OVMS_MODELS_DIR``, ...), set in the App's own
block of ``deploy.yaml`` (``apps.<component>.settings``) or in ``.env``.
``qp-ovms-config`` reads them when an OVMS App starts, and ``qp-calibrate
--ovms-models`` prints the two that pin a calibration. The streams and threads
are an embedder's or a reranker's; the two batching options are an LLM's, which
OVMS pulls and serves itself, uncalibrated. An App refuses the other kind's.

Example:
    >>> OvmsSettings().num_streams  # 0 = calibrated when the App starts
    0
    >>> OvmsSettings.variable_of("threads")
    'QP_OVMS_THREADS'
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..openvino_cpu_settings import OpenVinoCpuSettings

ENV_PREFIX = "QP_OVMS_"


class OvmsSettings(BaseSettings):
    """Every ``QP_OVMS_*`` variable Python reads, one field each.

    Construction checks them, so a bad pin stops an OVMS App before OVMS
    starts, with a message naming the variables and the values.

    Example:
        >>> OvmsSettings(num_streams=2, threads=8).threads
        8
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        # Between the block and its key: QP_OVMS_OPENVINO__HYPER_THREADING.
        env_nested_delimiter="__",
        # The launch script's own variables share the prefix (QP_OVMS_MODELS_DIR,
        # QP_OVMS_BUNDLE, QP_OVMS_PREFIX_PROXY) and are not read here.
        extra="ignore",
    )

    # Requests OVMS runs at once on its model (NUM_STREAMS); 0 = calibrated at
    # start-up, by qp-ovms-config.
    num_streams: int = Field(default=0, ge=0)
    # Threads its inference shares, all streams together (INFERENCE_NUM_THREADS);
    # 0 = calibrated at start-up, up to the CPUs.
    threads: int = Field(default=0, ge=0)
    # How long one OVMS start may take to answer before calibration fails.
    start_timeout_s: float = Field(default=120.0, gt=0)
    # OpenVINO's CPU properties of the graph's plugin_config, beside the two
    # counts above (QP_OVMS_OPENVINO__*): the block the in-process models read
    # as local_models.openvino.
    openvino: OpenVinoCpuSettings = Field(default_factory=OpenVinoCpuSettings)
    # An LLM's only: OVMS's own --dynamic_split_fuse and --max_num_batched_tokens.
    # Each flag is passed only away from OVMS's default (split fuse on; 0 = OVMS
    # picks the tokens per step), so the default command line is the one run on
    # a machine.
    dynamic_split_fuse: bool = True
    max_batched_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_ovms_tuning(self) -> OvmsSettings:
        """Fail at start-up when a pinned stream count exceeds its thread count.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If both are pinned and ``num_streams`` is above
                ``threads``: a stream needs a thread.
        """
        streams, threads = self.num_streams, self.threads
        if streams and threads and streams > threads:
            raise ValueError(
                f"{self.variable_of('num_streams')}={streams} exceeds "
                f"{self.variable_of('threads')}={threads}: a stream needs a "
                "thread; pin at most as many streams as threads"
            )
        return self

    def llm_flags(self) -> list[str]:
        """Return the ``ovms`` arguments of an LLM's two batching options.

        Returns:
            Each flag and its value, only for an option away from OVMS's own
            default; ``[]`` at both defaults.

        Example:
            >>> OvmsSettings(dynamic_split_fuse=False).llm_flags()
            ['--dynamic_split_fuse', 'false']
        """
        flags = []
        if not self.dynamic_split_fuse:
            flags += ["--dynamic_split_fuse", "false"]
        if self.max_batched_tokens:
            flags += ["--max_num_batched_tokens", str(self.max_batched_tokens)]
        return flags

    def require_no_llm_option(self, component: str) -> None:
        """Refuse an LLM's batching option on an embedder's or a reranker's App.

        Args:
            component: The embedder or reranker the App serves.

        Raises:
            ValueError: If either option was set, naming its variable.

        Example:
            >>> OvmsSettings().require_no_llm_option("reranker")
        """
        self._refuse_set(
            ("dynamic_split_fuse", "max_batched_tokens"),
            f"{component} is no LLM, and the option applies to an LLM served by "
            "OVMS only; unset it where it was set: this App's block of deploy.yaml "
            f"(apps.{component}.settings), its defaults.settings, or .env",
        )

    def require_no_stream_pin(self, component: str) -> None:
        """Refuse a pinned stream or thread count on an LLM's App.

        OVMS pulls and serves an LLM as it is: nothing is calibrated, and no
        stream or thread count would be applied.

        Args:
            component: The LLM the App serves.

        Raises:
            ValueError: If either count is pinned (above 0), naming its variable.

        Example:
            >>> OvmsSettings().require_no_stream_pin("llm")
        """
        pinned = [f for f in ("num_streams", "threads") if getattr(self, f)]
        self._refuse_set(
            tuple(pinned),
            f"{component} is an LLM, and OVMS serves it uncalibrated, so the count "
            "would be ignored; unset it in this App's block of deploy.yaml",
        )

    def _refuse_set(self, fields: tuple[str, ...], reason: str) -> None:
        found = [f for f in fields if f in self.model_fields_set]
        if found:
            # As the variable is written: a bool's value is "false", not "False".
            named = ", ".join(
                f"{self.variable_of(f)}={str(getattr(self, f)).lower()}" for f in found
            )
            raise ValueError(f"{named} is set, but {reason}")

    @classmethod
    def variable_of(cls, field: str) -> str:
        """Return the environment variable a field is read from.

        Args:
            field: The field name.

        Returns:
            ``QP_OVMS_<FIELD>``.

        Raises:
            ValueError: If ``field`` is not a field of these settings.

        Example:
            >>> OvmsSettings.variable_of("num_streams")
            'QP_OVMS_NUM_STREAMS'
        """
        if field not in cls.model_fields:
            raise ValueError(
                f"OvmsSettings has no field {field!r}; expected one of: "
                f"{', '.join(map(repr, cls.model_fields))}"
            )
        return ENV_PREFIX + field.upper()
