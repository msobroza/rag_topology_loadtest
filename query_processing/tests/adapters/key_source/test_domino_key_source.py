"""The ``domino`` key source told of refusals that arrive late.

Shown here, not at the workflow app: a late refusal of an older token needs two
calls in flight at once with the same token, which the app's requests cannot be
made to interleave on purpose. The token URL is an HTTP transport stand-in that
serves numbered tokens.
"""

from __future__ import annotations

import httpx

from query_processing.adapters.key_source.domino.domino_key_source import (
    DominoKeySource,
)
from query_processing.infrastructure.http.http_policy import HttpPolicy

TOKEN_URL = "http://domino.test/access-token"


class NumberedTokens:
    """Domino's token URL, serving ``token-1``, ``token-2``, ... in turn."""

    def __init__(self) -> None:
        self.served = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.served += 1
        return httpx.Response(200, text=f"token-{self.served}")


def test_a_late_refusal_of_the_old_token_keeps_the_newer_one() -> None:
    tokens = NumberedTokens()
    with httpx.Client(transport=httpx.MockTransport(tokens)) as client:
        source = DominoKeySource(
            client,
            token_url=TOKEN_URL,
            ttl_s=240.0,
            policy=HttpPolicy(timeout_s=5.0, retries=0, backoff_s=0.0),
        )
        old = source.key(may_ask=True)
        source.refused(old)  # the first call refused: the token is forgotten
        new = source.key(may_ask=True)
        source.refused(old)  # a second call, sent with the old token, refused late

        assert (old, new) == ("token-1", "token-2")
        assert source.key(may_ask=False) == "token-2"
    assert tokens.served == 2


def test_a_held_token_is_given_without_asking() -> None:
    tokens = NumberedTokens()
    with httpx.Client(transport=httpx.MockTransport(tokens)) as client:
        source = DominoKeySource(
            client,
            token_url=TOKEN_URL,
            ttl_s=240.0,
            policy=HttpPolicy(timeout_s=5.0, retries=0, backoff_s=0.0),
        )
        assert source.key(may_ask=False) is None  # none held: it would have to ask
        assert tokens.served == 0
        source.key(may_ask=True)
        assert source.key(may_ask=False) == "token-1"
    assert tokens.served == 1
