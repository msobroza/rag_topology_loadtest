"""A model server's pass-through of raw flags, beside its named settings.

Each model server (OVMS's ``QP_OVMS_ARGS``, vLLM's ``QP_VLLM_ARGS``) takes named
settings and one variable of raw flags for the rest. A flag given both ways is
refused rather than passed twice: a server reads the last one, or exits, and
neither tells which value ran. OVMS spells its flags with ``_`` and vLLM with
``-``, and both accept ``--flag value`` and ``--flag=value``, so the spellings
are read alike. Short spellings (vLLM's ``-O3``, ``-tp``) are the caller's.

Example:
    >>> words = split_pass_through("--rest-workers=8", variable="QP_OVMS_ARGS")
    >>> names_flag(words[0], "--rest_workers")
    True
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence


def split_pass_through(words: str, *, variable: str) -> list[str]:
    """Split a pass-through variable's text into words, as a shell would.

    Args:
        words: The pass-through variable's text, such as ``"--rest_workers 8"``.
        variable: The variable's name, for the error.

    Returns:
        The words, quotes removed.

    Raises:
        ValueError: If a quote is left open; it names ``variable`` and the text.

    Example:
        >>> split_pass_through("--log_level 'DEBUG'", variable="QP_OVMS_ARGS")
        ['--log_level', 'DEBUG']
    """
    try:
        return shlex.split(words)
    except ValueError as error:
        message = f"{variable}={words!r} is no list of shell words: {error}"
        raise ValueError(message) from error


def names_flag(word: str, flag: str) -> bool:
    """Whether ``word`` is ``flag`` or ``flag=value``, with ``_`` and ``-`` alike.

    Args:
        word: One word of a pass-through.
        flag: The flag a named setting gives, in its server's spelling.

    Returns:
        True if the word gives that flag.

    Example:
        >>> names_flag("--rest-workers=8", "--rest_workers")
        True
        >>> names_flag("--rest_workers_extra", "--rest_workers")
        False
    """
    name = word.split("=", 1)[0]
    return _spelling(name) == _spelling(flag)


def refuse_given_twice(
    named: Mapping[str, str], words: Sequence[str], *, pass_through: str
) -> None:
    """Refuse a flag a named setting gives that the pass-through gives too.

    Args:
        named: Each flag a named setting gives, mapped to that setting's variable
            and value, such as ``{"--rest_workers": "QP_OVMS_REST_WORKERS=4"}``.
        words: The pass-through's words, from ``split_pass_through``.
        pass_through: The pass-through variable's name, such as ``QP_OVMS_ARGS``.

    Raises:
        ValueError: If a word of ``words`` names a flag of ``named``; it holds
            the setting and the pass-through's word, and asks to state it once.

    Example:
        >>> refuse_given_twice({"--rest_workers": "QP_OVMS_REST_WORKERS=4"},
        ...                    ["--log_level", "INFO"], pass_through="QP_OVMS_ARGS")
    """
    for flag, setting in named.items():
        index = _index_of(words, flag)
        if index is not None:
            raise ValueError(
                f"{setting} gives {flag}, and {pass_through} gives it again as "
                f"{_given(words, index)!r}: state it once"
            )


def _index_of(words: Sequence[str], flag: str) -> int | None:
    return next((i for i, word in enumerate(words) if names_flag(word, flag)), None)


def _spelling(flag: str) -> str:
    # OVMS writes --rest_workers and vLLM --max-num-seqs, but a person may type
    # either spelling, so both name the same flag here.
    return flag.replace("_", "-")


def _given(words: Sequence[str], index: int) -> str:
    # The value of "--flag value" is the next word, unless that word is a flag.
    word = words[index]
    following = words[index + 1] if index + 1 < len(words) else ""
    if "=" in word or not following or following.startswith("-"):
        return word
    return f"{word} {following}"
