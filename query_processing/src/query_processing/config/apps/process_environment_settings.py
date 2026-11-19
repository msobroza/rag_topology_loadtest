"""What a process of ``qp-serve`` or ``qp-model-server`` needs in its environment.

The memory allocator and OpenMP's thread binding are read by the dynamic linker
and by the OpenMP runtime when the process starts, before any Python runs, so
setting them from Python is too late: the command restarts itself once with
them (``infrastructure/process_environment.py``). Flat ``QP_PROCESS_*``
variables, set per App as the ``QP_OVMS_*`` ones are. Each default is "not
set", so an App that sets nothing starts as it always has, with no restart.

Example:
    >>> ProcessEnvironmentSettings().wanted_environment()
    {}
    >>> environment_to_restart_with({"OMP_PROC_BIND": "close"}, {})
    {'OMP_PROC_BIND': 'close'}
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import SettingsConfigDict

from .base_app_settings import BaseAppSettings

ENV_PREFIX = "QP_PROCESS_"
PRELOAD = "LD_PRELOAD"
# The variables the OpenMP runtime reads, each the upper-cased name of the
# field that sets it, so the setting's variable is variable_of(its lower case).
OPENMP_VARIABLES = ("OMP_PROC_BIND", "KMP_AFFINITY")


class ProcessEnvironmentSettings(BaseAppSettings):
    """Every ``QP_PROCESS_*`` variable, one field each.

    Example:
        >>> ProcessEnvironmentSettings(omp_proc_bind="close").wanted_environment()
        {'OMP_PROC_BIND': 'close'}
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=".env", extra="ignore"
    )

    # The allocator put first in LD_PRELOAD, such as libjemalloc.so.2; "" = the
    # C library's. PyTorch's guide: jemalloc or TCMalloc "can get better
    # performance"; not measured under openvino (plan, decision 7).
    memory_allocator_library: str = ""
    # Under torch only: OpenVINO pins its own threads (cpu_pinning). "" = unset.
    omp_proc_bind: str = ""
    # Read only by Intel's OpenMP library, which the project does not install.
    kmp_affinity: str = ""

    @field_validator("memory_allocator_library")
    @classmethod
    def _library_is_a_file(cls, library: str) -> str:
        # Refused here, before the restart: a preload naming no file makes the
        # dynamic linker warn on every process and load nothing.
        if library and not Path(library).is_file():
            raise ValueError(
                f"{cls.variable_of('memory_allocator_library')}={library!r} names "
                "no file; expected the path of a library installed in the environment, "
                "such as /usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
            )
        return library

    def wanted_environment(self) -> dict[str, str]:
        """Return the variables the process must start with, only those set.

        Returns:
            ``LD_PRELOAD`` (the library alone), ``OMP_PROC_BIND`` and
            ``KMP_AFFINITY``, each when its setting is not ``""``.

        Example:
            >>> ProcessEnvironmentSettings(kmp_affinity="compact").wanted_environment()
            {'KMP_AFFINITY': 'compact'}
        """
        wanted = {PRELOAD: self.memory_allocator_library}
        wanted |= {v: getattr(self, v.lower()) for v in OPENMP_VARIABLES}
        return {variable: value for variable, value in wanted.items() if value}

    def restart_environment(self, environ: Mapping[str, str]) -> dict[str, str] | None:
        """Return the environment to restart with, or ``None`` when none is needed.

        Args:
            environ: The environment the process has.

        Returns:
            What ``environment_to_restart_with`` returns for these settings.

        Example:
            >>> ProcessEnvironmentSettings().restart_environment({}) is None
            True
        """
        return environment_to_restart_with(self.wanted_environment(), environ)


def environment_to_restart_with(
    wanted: Mapping[str, str], environ: Mapping[str, str]
) -> dict[str, str] | None:
    """Return ``environ`` with the wanted values, or ``None`` if it holds them all.

    ``None`` is what stops a second restart: the restarted process finds the
    library among the entries of ``LD_PRELOAD`` and the rest at their values.

    Args:
        wanted: What ``wanted_environment`` returns.
        environ: The environment the process has.

    Returns:
        A copy of ``environ`` with the OpenMP variables set and the library
        first in ``LD_PRELOAD``; ``None`` when nothing would change.

    Raises:
        ValueError: If ``environ`` holds an OpenMP variable at another value
            than its setting: given both ways.

    Example:
        >>> wanted, environ = {"LD_PRELOAD": "/l/j.so"}, {"LD_PRELOAD": "/a.so"}
        >>> environment_to_restart_with(wanted, environ)
        {'LD_PRELOAD': '/l/j.so:/a.so'}
    """
    _refuse_openmp_given_both_ways(wanted, environ)
    environment = dict(environ)
    environment.update({v: wanted[v] for v in OPENMP_VARIABLES if v in wanted})
    library = wanted.get(PRELOAD, "")
    preloaded = environ.get(PRELOAD, "")
    if library and library not in re.split(r"[: ]+", preloaded):
        environment[PRELOAD] = f"{library}:{preloaded}" if preloaded else library
    return None if environment == dict(environ) else environment


def _refuse_openmp_given_both_ways(
    wanted: Mapping[str, str], environ: Mapping[str, str]
) -> None:
    """Raise if an OpenMP variable is in ``environ`` at another value than wanted.

    Args:
        wanted: What ``wanted_environment`` returns.
        environ: The environment the process has.

    Raises:
        ValueError: Naming the setting and the variable, each with its value.
    """
    for variable in OPENMP_VARIABLES:
        named, given = wanted.get(variable), environ.get(variable)
        if named and given is not None and given != named:
            setting = ProcessEnvironmentSettings.variable_of(variable.lower())
            raise ValueError(
                f"{setting}={named} and {variable}={given} are given both ways; "
                f"expected {variable} only through {setting}: unset {variable}"
            )
