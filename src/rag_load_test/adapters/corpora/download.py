"""Download a corpus's files under one :class:`HttpPolicy` and convert them.

Each GET times out after ``timeout_s``; transport errors, 5xx and 429 are
retried ``retries`` times, waiting ``policy.backoff_before(attempt)`` in between;
any other status of 400 or more fails at once. Every file is held in memory
until all are fetched, so a failure writes nothing.

Example:
    >>> with httpx.Client() as client:
    ...     downloader = CorpusDownloader(client, policy)
    ...     fetched = downloader.fetch(CORPORA["quebec-insurance"])
    >>> len(fetched.questions)
    82
"""

from __future__ import annotations

import time
from collections.abc import Callable
from http import HTTPStatus
from typing import NamedTuple

import httpx

from ...domain.corpus import SourceDocument
from ...domain.errors import describe_error
from ..http.http_policy import HttpPolicy
from .catalog import CorpusDefinition
from .conversion import corpus_questions, reference_documents


class CorpusFetchError(Exception):
    """A corpus file that could not be downloaded.

    Args:
        url: The file's URL.
        detail: The HTTP status, or the transport error, of the last attempt.

    Example:
        >>> str(CorpusFetchError("https://x.test/a.jsonl", "HTTP 404"))
        'https://x.test/a.jsonl: HTTP 404'
    """

    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"{url}: {detail}")
        self.url = url


class FetchedCorpus(NamedTuple):
    """A corpus as ``rag-ingest`` and the load test read it."""

    documents: list[SourceDocument]
    questions: list[str]


class _Retryable(NamedTuple):
    # An attempt a retry may fix, and how to report it if none does.
    detail: str


class CorpusDownloader:
    """GET corpus files and convert them, retrying per one policy.

    Args:
        client: The HTTP client every GET goes through.
        policy: Timeout, retries and backoff of every GET.
        sleep: Waits between attempts; tests pass a recorder.

    Example:
        >>> downloader = CorpusDownloader(httpx.Client(), HttpPolicy(30.0, 2, 0.2))
    """

    def __init__(
        self,
        client: httpx.Client,
        policy: HttpPolicy,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client, self._policy, self._sleep = client, policy, sleep

    def fetch(self, corpus: CorpusDefinition) -> FetchedCorpus:
        """Download every file of ``corpus`` and convert it.

        Args:
            corpus: Which files to download, and from where.

        Returns:
            The documents of every reference file, in table then line order, and
            the questions.

        Example:
            >>> downloader.fetch(CORPORA["quebec-insurance"]).documents[0].id
            'A-25:00001'
        """
        documents = [
            doc
            for path in corpus.reference_paths
            for doc in reference_documents(path, self.get_text(corpus.url(path)))
        ]
        questions_text = self.get_text(corpus.url(corpus.questions_path))
        return FetchedCorpus(
            documents, corpus_questions(corpus.questions_path, questions_text)
        )

    def get_text(self, url: str) -> str:
        """GET ``url`` until an attempt succeeds or the retries run out.

        Args:
            url: The file to download.

        Returns:
            The body of the first answer below 400, as text.

        Raises:
            CorpusFetchError: On a status a retry cannot fix, or when every
                attempt failed; it names the URL and the status or error.

        Example:
            >>> downloader.get_text(corpus.url(corpus.questions_path))[:12]
            '{"question":'
        """
        last = _Retryable("no attempt made")
        for attempt in range(self._policy.retries + 1):
            if attempt:
                self._sleep(self._policy.backoff_before(attempt))
            outcome = self._attempt(url)
            if not isinstance(outcome, _Retryable):
                return outcome
            last = outcome
        raise CorpusFetchError(url, last.detail)

    def _attempt(self, url: str) -> str | _Retryable:
        # One GET: the body, a _Retryable, or a raised error for a 4xx no retry fixes.
        try:
            response = self._client.get(url, timeout=self._policy.timeout_s)
        except httpx.TransportError as exc:
            return _Retryable(describe_error(exc))
        status = response.status_code
        if status < HTTPStatus.BAD_REQUEST:
            return response.text
        retryable = status >= HTTPStatus.INTERNAL_SERVER_ERROR
        if retryable or status == HTTPStatus.TOO_MANY_REQUESTS:
            return _Retryable(f"HTTP {status}")
        raise CorpusFetchError(url, f"HTTP {status}")
