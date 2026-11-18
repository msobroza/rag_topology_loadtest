"""A route, and the router's request and answer: a vector in, a route choice out.

A use case states its routes in ``routes.yaml``, a catalogue: each route has a
name, a description of what it means, and the utterances that exemplify it. A
router lists some of them by name, and chooses one for each query, with its
confidence (spec 8).

Example:
    >>> Route(name="self_contained", description="Understood alone.",
    ...       utterances=["What does my policy cover?"]).name
    'self_contained'
    >>> RouteChoice(name="self_contained", confidence=0.9).name
    'self_contained'
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Route(_FrozenStrictModel):
    """One answer a router can choose: its name, what it means, its utterances."""

    name: str = Field(min_length=1)
    description: str
    # The messages a router learns the route from: one at least.
    utterances: list[str] = Field(min_length=1)


class RouteRequest(_FrozenStrictModel):
    """A query's vector, as the embedder the router reads gives it."""

    embedding: list[float]


class RouteChoice(_FrozenStrictModel):
    """The route a router chose for one query, with its confidence."""

    name: str
    # Not bounded here: RouterContract checks that a router keeps it in [0, 1].
    confidence: float
