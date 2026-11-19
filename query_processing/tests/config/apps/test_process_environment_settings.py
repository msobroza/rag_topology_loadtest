"""What a process of ``qp-serve`` or ``qp-model-server`` needs in its environment.

The settings name the allocator to preload and OpenMP's binding; the
environment a restart needs is worked out from them and the one the process
has, and is ``None`` once the process already holds every wanted value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from query_processing.config.apps.process_environment_settings import (
    ProcessEnvironmentSettings,
    environment_to_restart_with,
)


def settings(**values: str) -> ProcessEnvironmentSettings:
    """The settings from ``values`` alone, with no ``.env`` read."""
    return ProcessEnvironmentSettings(_env_file=None, **values)  # type: ignore[call-arg]


def test_nothing_set_wants_nothing_and_restarts_nothing() -> None:
    assert settings().wanted_environment() == {}
    assert environment_to_restart_with({}, {"PATH": "/bin"}) is None


def test_each_setting_names_its_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libjemalloc.so.2"
    library.touch()
    monkeypatch.setenv("QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY", str(library))
    monkeypatch.setenv("QP_PROCESS_OMP_PROC_BIND", "close")
    monkeypatch.setenv("QP_PROCESS_KMP_AFFINITY", "granularity=fine,compact")

    assert settings().wanted_environment() == {
        "LD_PRELOAD": str(library),
        "OMP_PROC_BIND": "close",
        "KMP_AFFINITY": "granularity=fine,compact",
    }


def test_the_allocator_goes_first_in_ld_preload() -> None:
    environment = environment_to_restart_with(
        {"LD_PRELOAD": "/l/libjemalloc.so.2"}, {"LD_PRELOAD": "/l/a.so", "PATH": "/b"}
    )

    assert environment == {"LD_PRELOAD": "/l/libjemalloc.so.2:/l/a.so", "PATH": "/b"}


def test_the_allocator_alone_when_nothing_was_preloaded() -> None:
    environment = environment_to_restart_with({"LD_PRELOAD": "/l/j.so"}, {})

    assert environment == {"LD_PRELOAD": "/l/j.so"}


@pytest.mark.parametrize("preloaded", ["/l/j.so:/l/a.so", "/l/a.so /l/j.so"])
def test_a_process_already_in_its_environment_does_not_restart_again(
    preloaded: str,
) -> None:
    wanted = {"LD_PRELOAD": "/l/j.so", "OMP_PROC_BIND": "close"}
    environ = {"LD_PRELOAD": preloaded, "OMP_PROC_BIND": "close"}

    assert environment_to_restart_with(wanted, environ) is None


def test_openmp_binding_is_exported_when_the_environment_lacks_it() -> None:
    environment = environment_to_restart_with({"OMP_PROC_BIND": "close"}, {})

    assert environment == {"OMP_PROC_BIND": "close"}


@pytest.mark.parametrize("variable", ["OMP_PROC_BIND", "KMP_AFFINITY"])
def test_openmp_binding_given_both_ways_is_refused_naming_both(variable: str) -> None:
    with pytest.raises(
        ValueError, match=rf"QP_PROCESS_{variable}=close.*{variable}=spread"
    ):
        environment_to_restart_with({variable: "close"}, {variable: "spread"})


def test_a_library_that_is_no_file_is_refused_naming_its_variable_and_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.so"

    with pytest.raises(ValidationError) as refused:
        settings(memory_allocator_library=str(missing))

    assert "QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY" in str(refused.value)
    assert str(missing) in str(refused.value)
