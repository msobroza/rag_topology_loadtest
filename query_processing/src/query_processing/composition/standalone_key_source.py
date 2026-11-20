"""The key source of a command that reads no workflow file, from its ``key`` block.

The workflow app opens its key sources from its settings (``HttpResources``).
A command that reads no workflow file, such as the load-test client, has a
``key`` block of its own and is opened its source here, the one place outside
the workflow app that builds a key source. It is given what the settings of
the workflow app would give: the variable the block is read from, for the
errors; the ``domino`` source's token URL and the time it holds a token; the
timeout, retries and backoff of asking, its own, since it has no ``http``
block; and the ``.env`` file an ``env`` source also reads. It refuses an
unknown source and a key of the block its source does not read with the lines
start-up prints, and closes what it opened when the block ends. A command with
no ``key`` block that sends the run token, ``qp-deploy status``, is opened its
``domino`` source here too (``run_token_source``).

Example:
    >>> needs = StandaloneKeyNeeds("QP_LOADTEST_KEY", token_url=url,
    ...     token_ttl_s=240.0, timeout_s=60.0, retries=2, backoff_s=0.2,
    ...     env_file=Path(".env"))
    >>> with opened_standalone_key_source(block, needs) as source:
    ...     source.key(may_ask=True)
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..adapters.key_source.base_key_source import KEY_SOURCES, BaseKeySource
from ..adapters.key_source.domino.domino_key_source import DominoKeySource
from ..config.components.key_settings import KeySettings
from ..config.shared_settings import DefaultHttpSettings, DominoSettings
from ..infrastructure.http.http_policy import HttpPolicy


@dataclass(frozen=True)
class StandaloneKeyNeeds:
    """What a key source is opened with when no workflow file gives it.

    Example:
        >>> StandaloneKeyNeeds("QP_LOADTEST_KEY", token_url=url, token_ttl_s=240.0,
        ...     timeout_s=60.0, retries=2, backoff_s=0.2, env_file=None).variable
        'QP_LOADTEST_KEY'
    """

    variable: str  # the block's variable, such as QP_LOADTEST_KEY, for errors
    token_url: str  # where the domino source asks for the run token
    token_ttl_s: float  # how long it holds one
    timeout_s: float  # of one ask
    retries: int  # asks after the first
    backoff_s: float  # before the first retry; doubles
    env_file: Path | None  # the .env file an env source also reads


@contextmanager
def opened_standalone_key_source(
    block: KeySettings,
    needs: StandaloneKeyNeeds,
    transport: httpx.BaseTransport | None = None,
) -> Generator[BaseKeySource, None, None]:
    """Open the key source a ``key`` block names, and close it when the block ends.

    Args:
        block: The ``key`` block.
        needs: The variable, the domino source's settings, the policy of
            asking and the ``.env`` file.
        transport: The domino source's transport; None for the network
            (tests pass an ``httpx.MockTransport``).

    Yields:
        The opened source.

    Raises:
        ValueError: An unknown source, or a key of the block its source does
            not read, naming the variable as start-up does.
        MissingKeyError: If the source finds no key, such as an ``env``
            variable unset or empty.

    Example:
        >>> with opened_standalone_key_source(KeySettings(source="domino"),
        ...                                   needs) as source:
        ...     source.key(may_ask=True)
    """
    source = checked_key_source(block, needs.variable)
    if source is not DominoKeySource:
        yield source.from_block(block.read_keys(), env_file=needs.env_file)
        return
    with httpx.Client(timeout=needs.timeout_s, transport=transport) as client:
        yield _domino_source(client, needs)


def checked_key_source(block: KeySettings, variable: str) -> type[BaseKeySource]:
    """Return the class of the source a block names, once the block is one it reads.

    The one declaration of what start-up refuses of a ``key`` block: the
    check of registered names calls it for every ``key`` block of the
    workflow settings, and a command that reads no workflow file through
    ``opened_standalone_key_source``. The keys beside ``source`` must be the
    ones its source reads, each and no other.

    Args:
        block: The ``key`` block.
        variable: The variable it is read from, such as
            ``QP_COMPONENTS__LLM__API__KEY`` or ``QP_LOADTEST_KEY``.

    Returns:
        The registered key source.

    Raises:
        ValueError: ``<variable>__SOURCE: unknown key source ...`` naming the
            known ones, or ``<variable>: the 'env' key source reads 'variable'
            beside 'source'; got none``.

    Example:
        >>> checked_key_source(KeySettings(source="env", variable="X"),
        ...                    "QP_LOADTEST_KEY").__name__
        'EnvKeySource'
    """
    try:
        source = KEY_SOURCES.get(block.source)
    except ValueError as exc:
        raise ValueError(f"{variable}__SOURCE: {exc}") from exc
    reads, given = source.reads, sorted(block.read_keys())
    if given != sorted(reads):
        expected = ", ".join(map(repr, reads)) or "no key"
        got = ", ".join(map(repr, given)) or "none"
        raise ValueError(
            f"{variable}: the {block.source!r} key source reads {expected} beside "
            f"'source'; got {got}"
        )
    return source


def run_token_source(
    client: httpx.Client, *, token_url: str, timeout_s: float, retries: int
) -> BaseKeySource:
    """Open the ``domino`` source of a command with neither workflow file nor key block.

    ``qp-deploy status`` asks for the run token it sends to the workflow App
    this way, so one piece of code asks Domino for a token. It asks through the
    command's own client, under the command's timeout and retries; between
    them it waits the shared ``http`` block's default backoff, and holds a
    token the ``domino`` block's default time, which one ask never reaches.

    Args:
        client: The command's client, which the token is asked for through.
        token_url: Where the run serves its token.
        timeout_s: The wait for one answer.
        retries: Asks after the first, on what the source retries.

    Returns:
        The source, which asks at its first ``key(may_ask=True)``.

    Example:
        >>> run_token_source(http, token_url=url, timeout_s=30.0,
        ...                  retries=2).key(may_ask=True)
        'eyJ...'
    """
    policy = HttpPolicy(timeout_s, retries, DefaultHttpSettings().backoff_s)
    return domino_key_source(client, DominoSettings(token_url=token_url), policy)


def domino_key_source(
    client: httpx.Client, domino: DominoSettings, policy: HttpPolicy
) -> DominoKeySource:
    """Build the ``domino`` key source: the one place the composition builds it.

    The workflow app's (``HttpResources``), a command's with a ``key`` block
    and ``qp-deploy status``'s are each built here, from a ``domino`` block.

    Args:
        client: The blocking client the token is asked for through.
        domino: The token's URL and the time a token is held.
        policy: The timeout, retries and backoff of asking.

    Returns:
        The source, which asks at its first ``key(may_ask=True)``.

    Example:
        >>> domino_key_source(httpx.Client(), DominoSettings(), policy)
    """
    return DominoKeySource(
        client, token_url=domino.token_url, ttl_s=domino.token_ttl_s, policy=policy
    )


def _domino_source(client: httpx.Client, needs: StandaloneKeyNeeds) -> DominoKeySource:
    policy = HttpPolicy(needs.timeout_s, needs.retries, needs.backoff_s)
    domino = DominoSettings(token_url=needs.token_url, token_ttl_s=needs.token_ttl_s)
    return domino_key_source(client, domino, policy)
