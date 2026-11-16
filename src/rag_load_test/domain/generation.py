"""One answer of the LLM port: its text and the token counts behind it.

The counts are what the endpoint reported, not a count of our own: a load test
reads tokens per second from them, so a guess would mislead it.

Example:
    >>> Generation("Paris.", prompt_tokens=829, completion_tokens=128).text
    'Paris.'
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Generation:
    """An answer and its token counts; a count is None when none was reported.

    Attributes:
        text: The answer text; "" when the answer carried no content.
        prompt_tokens: The tokens of the conversation sent; None: not reported.
        completion_tokens: The tokens of the answer; None: not reported.

    Example:
        >>> Generation("Paris.") == Generation("Paris.", None, None)
        True
    """

    text: str
    prompt_tokens: int | None = None  # None: the endpoint reported none
    completion_tokens: int | None = None
