"""An OVMS model server App's own settings: its streams, its threads, its start.

OpenVINO Model Server serves one component per App, on that App's hardware, so
its stream and thread counts are the App's, not the workflow's: they are no
value of the workflow file. They are flat ``QP_OVMS_*`` variables, beside the
ones the launch script reads (``QP_OVMS_MODELS_DIR``, ...), set in the App's own
block of ``deploy.yaml`` (``apps.<component>.settings``) or in ``.env``.
``qp-ovms-config`` reads them when an OVMS App starts, and ``qp-calibrate
--ovms-models`` prints the two that pin a calibration.

Example:
    >>> OvmsSettings().num_streams  # 0 = calibrated when the App starts
    0
    >>> OvmsSettings.variable_of("threads")
    'QP_OVMS_THREADS'
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
