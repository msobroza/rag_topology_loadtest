"""``OvmsSettings``: a server flag named or passed through ``QP_OVMS_ARGS``.

Either way gives the same command line. A named setting left at its default
yields to the pass-through; one that was set, or a flag the launch script
gives, is refused beside it, naming both.
"""

from __future__ import annotations

import os

import pytest

from query_processing.config.apps.ovms_flags import LLM_FLAGS, SERVER_FLAGS, spelled
from query_processing.config.apps.ovms_settings import OvmsSettings
from rag_components import shell_settings


@pytest.fixture(autouse=True)
def _no_ovms_variable_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in shell_settings(os.environ, ("QP_OVMS_",)):
        monkeypatch.delenv(name)


def ovms_settings(**values: object) -> OvmsSettings:
    """The settings without reading a stray ``.env``."""
    return OvmsSettings(_env_file=None, **values)  # type: ignore[arg-type]


def command_line(settings: OvmsSettings) -> list[str]:
    """An LLM's App's flags, as ``qp-ovms-config --serve-llm`` prints them."""
    return settings.llm_flags() + settings.server_flags()


def as_ovms_reads(words: list[str]) -> dict[str, str]:
    """Each flag and the value ovms runs on, a flag at OVMS's own value left out.

    A flag written twice fails: ovms would read one of two values.
    """
    own = {f.flag: spelled(f.ovms_own) for f in (*SERVER_FLAGS, *LLM_FLAGS)}
    read: dict[str, str] = {}
    for index, word in enumerate(words):
        if not word.startswith("--"):
            continue
        following = words[index + 1] if index + 1 < len(words) else "--"
        flag, value = word, "true" if following.startswith("--") else following
        assert flag not in read, f"{flag} is written twice in {words}"
        read[flag] = value
    return {flag: value for flag, value in read.items() if own.get(flag) != value}


@pytest.mark.parametrize(
    ("named", "passed"),
    [
        ({"rest_workers": 4}, "--rest_workers 4"),
        ({"metrics": True}, "--metrics_enable"),
        ({"log_level": "DEBUG"}, "--log_level DEBUG"),
        ({"cache_dir": "/cache"}, "--cache_dir /cache"),
        ({"dynamic_split_fuse": True}, "--dynamic_split_fuse true"),
        ({"max_batched_tokens": 4096}, "--max_num_batched_tokens 4096"),
    ],
)
def test_a_server_flag_named_or_passed_through_gives_the_same_command_line(
    named: dict[str, object], passed: str
) -> None:
    # Compared as ovms reads them: split fuse set to OVMS's own value writes
    # nothing by name, and "--dynamic_split_fuse true" when passed through.
    assert as_ovms_reads(command_line(ovms_settings(**named))) == (
        as_ovms_reads(command_line(ovms_settings(args=passed)))
    )


@pytest.mark.parametrize(
    ("passed", "expected"),
    [
        (
            "--dynamic_split_fuse true",
            ["--max_num_batched_tokens", "8192", "--dynamic_split_fuse", "true"],
        ),
        (
            "--max_num_batched_tokens=4096",
            ["--dynamic_split_fuse", "false", "--max_num_batched_tokens=4096"],
        ),
    ],
)
def test_an_llm_flag_left_at_its_default_yields_to_the_pass_through(
    passed: str, expected: list[str]
) -> None:
    # The pass-through is then the one statement of the flag, as a property
    # of the OpenVINO block left unset yields to its properties.
    assert command_line(ovms_settings(args=passed)) == expected


@pytest.mark.parametrize(
    ("named", "passed", "setting"),
    [
        (
            {"dynamic_split_fuse": False},
            "--dynamic_split_fuse true",
            "QP_OVMS_DYNAMIC_SPLIT_FUSE=false",
        ),
        (
            {"max_batched_tokens": 8192},
            "--max_num_batched_tokens 4096",
            "QP_OVMS_MAX_BATCHED_TOKENS=8192",
        ),
    ],
)
def test_an_llm_flag_set_and_passed_through_is_refused_naming_both(
    named: dict[str, object], passed: str, setting: str
) -> None:
    with pytest.raises(ValueError, match="QP_OVMS_ARGS") as raised:
        ovms_settings(**named, args=passed)

    assert setting in str(raised.value)
    assert repr(passed) in str(raised.value)


@pytest.mark.parametrize(
    ("named", "passed"),
    [
        ({"rest_workers": 4}, "--rest-workers=8"),
        ({"log_level": "INFO"}, "--log_level DEBUG"),
        ({"metrics": True}, "--metrics_enable"),
        ({"max_num_seqs": 64}, "--max_num_seqs=32"),
        ({"dynamic_split_fuse": False}, "--dynamic_split_fuse true"),
        ({}, "--rest_port 9000"),
        ({}, "--config_path /x.json"),
        ({}, "--source-model=org/m"),
        ({}, "--task text_generation"),
    ],
)
def test_a_flag_given_both_ways_or_owned_by_the_script_is_refused(
    named: dict[str, object], passed: str
) -> None:
    with pytest.raises(ValueError, match="QP_OVMS_ARGS") as raised:
        ovms_settings(**named, args=passed)

    assert passed.split()[0] in str(raised.value)


def test_a_pass_through_with_an_open_quote_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="QP_OVMS_ARGS="):
        ovms_settings(args="--cache_dir '/a b")


def test_sequences_are_not_checked_against_a_default_the_pass_through_replaced() -> (
    None
):
    # The default 8192 yields to the pass-through's 16384: nobody set it.
    settings = ovms_settings(max_num_seqs=9000, args="--max_num_batched_tokens 16384")

    assert command_line(settings)[-2:] == ["--max_num_batched_tokens", "16384"]
