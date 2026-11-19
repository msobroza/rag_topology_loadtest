"""Tests of ``read_pages``: a Domino list read page after page, over a mock transport."""

from __future__ import annotations

import httpx
import pytest

from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.domino_lookups.paged_list import read_pages


class OneItemThenNone:
    """Answers one item at offset 0 and none after, counting the requests."""

    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata, self.queries = metadata, list[dict[str, str]]()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.queries.append(dict(request.url.params))
        offset = request.url.params.get("offset")
        items = [{"id": "only"}] if offset == "0" else []
        return httpx.Response(200, json={"items": items, "metadata": self.metadata})


class SignInPage:
    """Answers 200 with a page that is not JSON, as a proxy asking to sign in does."""

    def __call__(self, _request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>Sign in</html>")


class SamePageWhateverTheOffset:
    """Answers the same page whatever the offset, with no total, counting the calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls > 10:  # a loop that never ends fails here, not in memory
            raise AssertionError("read_pages asked more than 10 pages")
        return httpx.Response(200, json={"items": [{"id": "e"}, {"id": "f"}]})


@pytest.mark.parametrize("metadata", [{"totalCount": 3}, {}])
def test_reading_stops_at_an_empty_page(metadata: dict[str, object]) -> None:
    answer = OneItemThenNone(metadata)
    http = httpx.Client(base_url="http://domino", transport=httpx.MockTransport(answer))
    assert read_pages(http, "/api/x", "items", needs="n") == [{"id": "only"}]
    assert answer.queries == [{"offset": "0"}, {"offset": "1"}]  # no limit


def test_an_answer_that_is_not_json_is_refused() -> None:
    page = httpx.MockTransport(SignInPage())
    http = httpx.Client(base_url="http://domino", transport=page)
    with pytest.raises(DominoDeployError) as refusal:
        read_pages(http, "/api/x", "items", needs="n")
    assert str(refusal.value) == (
        "Domino API GET /api/x answered HTTP 200 with a body that is not JSON: "
        "<html>Sign in</html>"
    )


def test_a_list_that_repeats_its_page_whatever_the_offset_is_refused() -> None:
    answer = SamePageWhateverTheOffset()
    http = httpx.Client(base_url="http://domino", transport=httpx.MockTransport(answer))
    with pytest.raises(DominoDeployError) as refusal:
        read_pages(http, "/api/x", "items", needs="n")
    assert str(refusal.value) == (
        "Domino API GET /api/x answered the same 2 items at offset 2 as at offset 0: "
        "the list repeats whatever the offset, so it cannot be read to its end"
    )
    assert answer.calls == 2
