"""The corpora ``rag-fetch-corpus`` knows: one ``CorpusDefinition`` each.

A second corpus is one more entry in ``CORPORA``; the CLI's corpus argument is a
choice over its keys. Where a corpus lives and what it is made of belong to its
definition, so they are written here rather than read from settings.

Example:
    >>> corpus = CORPORA["quebec-insurance"]
    >>> corpus.url(corpus.questions_path)
    'https://raw.githubusercontent.com/GRAAL-Research/.../QnA/V1/questions-answers.jsonl'
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusDefinition:
    """Where a corpus's files live and which ones make it up.

    Each reference file is JSON lines ``{"text", "url"?}``, one passage per line;
    its stem names its documents. The questions file is JSON lines with a
    ``question`` key; any other key (answers, criteria) is evaluation material.
    """

    name: str
    base_url: str
    reference_paths: tuple[str, ...]  # relative to base_url
    questions_path: str  # relative to base_url
    licence_note: str  # printed by every fetch

    def url(self, path: str) -> str:
        """Return the URL of one of the corpus's files.

        Args:
            path: The file's path relative to ``base_url``.

        Returns:
            ``base_url`` and ``path`` joined by one slash.

        Example:
            >>> CORPORA["quebec-insurance"].url("References/V1/AMF.jsonl")
            'https://raw.githubusercontent.com/.../main/References/V1/AMF.jsonl'
        """
        return f"{self.base_url}/{path}"


_QUEBEC_STEMS = (
    "A-25",
    "AMF",
    "CHAD",
    "GAA",
    "code_securite_routiere",
    "educaloi",
    "infoassurances",
)

# Beauchemin et al., NLLP 2024: French passages on Quebec automobile insurance and
# 82 questions on them. The repository states no licence, so nothing of it is
# committed here and every fetch says so.
QUEBEC_INSURANCE = CorpusDefinition(
    name="quebec-insurance",
    base_url="https://raw.githubusercontent.com/GRAAL-Research/quebec-insurance-rag-corpora/main",
    reference_paths=tuple(f"References/V1/{stem}.jsonl" for stem in _QUEBEC_STEMS),
    questions_path="QnA/V1/questions-answers.jsonl",
    licence_note=(
        "the Quebec automobile insurance corpus of GRAAL-Research states no formal "
        "licence; it is a publicly released research corpus, for research use only, "
        "to be cited as Beauchemin et al., NLLP 2024. Do not commit it."
    ),
)

CORPORA: dict[str, CorpusDefinition] = {c.name: c for c in (QUEBEC_INSURANCE,)}
