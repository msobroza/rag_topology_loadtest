"""An API format: a component type's request and answer to the wire and back.

The toolkit builds one that mirrors the request and the answer for a type that
registers none, and it refuses on the wire what the domain class refuses.
"""

from __future__ import annotations

import pytest

from query_processing.adapters.vector_store.base_vector_store import BaseVectorStore
from query_processing.domain.endpoints.api_format import api_format_of
from shout_extension.shouting import BaseShouter, Shouted, ShoutRequest
from shout_extension.whispering import BaseWhisperer, WordsFormat


def test_the_built_format_mirrors_the_request_and_the_answer() -> None:
    api_format = api_format_of(BaseShouter)

    assert api_format.request_to_wire(ShoutRequest(text="a")) == {"text": "a"}
    assert api_format.request_from_wire({"text": "a"}) == ShoutRequest(text="a")
    assert api_format.answer_to_wire(Shouted(text="A")) == {"text": "A"}
    assert api_format.answer_from_wire({"text": "A"}) == Shouted(text="A")


def test_the_built_format_refuses_a_body_of_another_shape() -> None:
    api_format = api_format_of(BaseShouter)

    with pytest.raises(ValueError, match="text"):
        api_format.request_from_wire({"txt": "a"})
    with pytest.raises(ValueError, match="Shouted"):
        api_format.answer_from_wire(["A"])


def test_the_built_format_refuses_what_the_request_refuses() -> None:
    api_format = api_format_of(BaseVectorStore)
    body = {"text": "q", "top_k": 2, "retrieval_mode": "dense"}

    with pytest.raises(ValueError, match="retrieval_mode='dense' needs embedding"):
        api_format.request_from_wire(body)


def test_the_built_format_sends_json_values() -> None:
    api_format = api_format_of(BaseVectorStore)
    request = api_format.request_from_wire(
        {"embedding": [0.5], "top_k": 2, "retrieval_mode": "dense"}
    )

    assert api_format.request_to_wire(request) == {
        "embedding": [0.5],
        "text": None,
        "top_k": 2,
        "retrieval_mode": "dense",
        "rrf_k": None,
        "metadata_filter": [],
    }


def test_a_format_registered_for_the_component_type_is_used() -> None:
    api_format = api_format_of(BaseWhisperer)

    assert isinstance(api_format, WordsFormat)
    assert api_format.request_to_wire(ShoutRequest(text="A")) == {"words": "A"}
