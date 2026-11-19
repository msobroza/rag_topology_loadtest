"""OpenVINO's CPU properties a site may choose: one block, read from two homes.

The in-process models read it as the workflow file's ``local_models.openvino``
(``QP_LOCAL_MODELS__OPENVINO__*``); an OVMS App reads it as its own
``QP_OVMS_OPENVINO__*`` (``apps/ovms_settings.py``) and writes it into its
graph's ``plugin_config``. Each property has a named setting, and any other
goes through ``properties``, as OpenVINO spells it. A setting left at its
default writes nothing, so OpenVINO's own value holds, except
``hyper_threading``, which this project measured. The stream and thread counts
are calibration's and never come from here.

Example:
    >>> OpenVinoCpuSettings().chosen()
    {'ENABLE_HYPER_THREADING': 'YES'}
    >>> OpenVinoCpuSettings(cpu_pinning=False).chosen()["ENABLE_CPU_PINNING"]
    'NO'
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .components.settings_blocks import SettingsBlock

# Each named setting and the property OpenVINO reads it as.
_PROPERTY_OF_FIELD: dict[str, str] = {
    "performance_hint": "PERFORMANCE_HINT",
    "performance_hint_num_requests": "PERFORMANCE_HINT_NUM_REQUESTS",
    "hyper_threading": "ENABLE_HYPER_THREADING",
    "cpu_pinning": "ENABLE_CPU_PINNING",
    "cpu_reservation": "ENABLE_CPU_RESERVATION",
    "inference_precision_hint": "INFERENCE_PRECISION_HINT",
    "execution_mode_hint": "EXECUTION_MODE_HINT",
}
# Calibration chooses these for each model; a pass-through would hide its choice.
_CALIBRATED_PROPERTIES = ("NUM_STREAMS", "INFERENCE_NUM_THREADS")


class OpenVinoCpuSettings(SettingsBlock):
    """The ``openvino`` block: OpenVINO CPU properties, named or passed through.

    Example:
        >>> OpenVinoCpuSettings(properties={"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}
        ...                     ).chosen()["SCHEDULING_CORE_TYPE"]
        'PCORE_ONLY'
    """

    # "" = not written: OpenVINO's own is LATENCY, one stream.
    performance_hint: Literal["", "LATENCY", "THROUGHPUT"] = ""
    # Caps the requests the THROUGHPUT hint creates; 0 = not written.
    performance_hint_num_requests: int = Field(default=0, ge=0)
    # In its latency mode OpenVINO keeps to physical cores: asked for 2 threads
    # on one core with two hyper-threads, it ran 1 and left the second CPU idle.
    # Allowed the hyper-threads it ran 2, and a rerank request took 2.94 s
    # instead of 3.38 (GCP sitting of 2026-11-15). So true, against OpenVINO's
    # own default, and always written.
    hyper_threading: bool = True
    # None = not written: OVMS and OpenVINO decide (off under a CPU quota).
    cpu_pinning: bool | None = None
    cpu_reservation: bool | None = None
    # "" = not written: OpenVINO already picks bf16 on a CPU with AMX.
    inference_precision_hint: Literal["", "f32", "bf16"] = ""
    # "" = not written: OpenVINO's own is PERFORMANCE.
    execution_mode_hint: Literal["", "PERFORMANCE", "ACCURACY"] = ""
    # Any other property, as OpenVINO names and spells it.
    properties: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_openvino_properties(self) -> OpenVinoCpuSettings:
        """Refuse a property stated twice, one calibration owns, or a lone count.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If a key of ``properties`` is also set by name, naming
                both; if it holds ``NUM_STREAMS`` or ``INFERENCE_NUM_THREADS``;
                or if a request count is given without the ``THROUGHPUT`` hint.
        """
        self._refuse_given_twice()
        self._refuse_calibrated()
        chosen = self.chosen()
        hint = chosen.get("PERFORMANCE_HINT", "")
        if "PERFORMANCE_HINT_NUM_REQUESTS" in chosen and hint != "THROUGHPUT":
            raise ValueError(
                "PERFORMANCE_HINT_NUM_REQUESTS="
                f"{chosen['PERFORMANCE_HINT_NUM_REQUESTS']} only caps what the "
                f"THROUGHPUT hint creates, but PERFORMANCE_HINT is {hint!r}: set "
                "performance_hint to THROUGHPUT, or leave the count at 0"
            )
        return self

    def chosen(self) -> dict[str, str]:
        """Return the properties to write, OpenVINO's names and spellings.

        Returns:
            Each named setting that writes a value, in the fields' order, then
            ``properties``. A named setting left unset whose property
            ``properties`` gives yields to it: the pass-through is then the
            one statement of it.

        Example:
            >>> OpenVinoCpuSettings(properties={"ENABLE_HYPER_THREADING": "NO"}
            ...                     ).chosen()
            {'ENABLE_HYPER_THREADING': 'NO'}
        """
        named = {
            _PROPERTY_OF_FIELD[field]: written
            for field in _PROPERTY_OF_FIELD
            if (written := self._written(field)) is not None
            and _PROPERTY_OF_FIELD[field] not in self.properties
        }
        return {**named, **self.properties}

    def _written(self, field: str) -> str | None:
        value = getattr(self, field)
        if isinstance(value, bool):
            return "YES" if value else "NO"
        if value in (None, "", 0):  # the "not set" of each field
            return None
        return str(value)

    def _refuse_given_twice(self) -> None:
        for field, name in _PROPERTY_OF_FIELD.items():
            if field in self.model_fields_set and name in self.properties:
                raise ValueError(
                    f"{field}={getattr(self, field)!r} and properties "
                    f"{name}={self.properties[name]!r} both set {name}: state it "
                    f"once, by {field} or in properties"
                )

    def _refuse_calibrated(self) -> None:
        for name in _CALIBRATED_PROPERTIES:
            if name in self.properties:
                raise ValueError(
                    f"properties holds {name}={self.properties[name]!r}, which "
                    "calibration chooses: remove it, or pin the count with the "
                    "component's or the App's own setting (its threads or streams)"
                )
