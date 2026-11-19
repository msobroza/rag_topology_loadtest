"""``BaseAppSettings.variable_of``: one rule for every flat App settings class.

Each class of ``config/apps/`` reads flat variables, its own prefix and the
upper-cased field, so a field's variable is worked out once, from the prefix.
"""

from __future__ import annotations

import pytest

from query_processing.config.apps.base_app_settings import BaseAppSettings
from query_processing.config.apps.ovms_settings import OvmsSettings
from query_processing.config.apps.process_environment_settings import (
    ProcessEnvironmentSettings,
)
from query_processing.config.apps.uvicorn_settings import UvicornSettings


@pytest.mark.parametrize(
    ("settings", "field", "variable"),
    [
        (OvmsSettings, "num_streams", "QP_OVMS_NUM_STREAMS"),
        (UvicornSettings, "timeout_keep_alive_s", "QP_UVICORN_TIMEOUT_KEEP_ALIVE_S"),
        (ProcessEnvironmentSettings, "omp_proc_bind", "QP_PROCESS_OMP_PROC_BIND"),
    ],
)
def test_a_field_is_read_from_its_prefix_and_its_name(
    settings: type[BaseAppSettings], field: str, variable: str
) -> None:
    assert issubclass(settings, BaseAppSettings)
    assert settings.variable_of(field) == variable


@pytest.mark.parametrize(
    "settings", [OvmsSettings, UvicornSettings, ProcessEnvironmentSettings]
)
def test_a_name_that_is_no_field_is_refused_naming_the_class_and_the_name(
    settings: type[BaseAppSettings],
) -> None:
    with pytest.raises(ValueError, match=f"{settings.__name__} has no field 'nope'"):
        settings.variable_of("nope")
