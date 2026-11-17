"""ROUGE-L between two answers an LLM wrote: the one module importing ``rouge-score``.

An LLM asked twice at temperature 0 can still word an answer otherwise, so
``qp-compare-answers`` holds its text with a metric computed the same way every
time, never text for text. The library comes with the toolkit's ``loadtest``
extra, which the Apps never install: ``qp-compare-answers`` imports this module
only when the answers' text is compared.

The library's own tokenizer keeps only ``a-z0-9`` and so drops accented
letters ("L'été dernier" would be ``l``, ``t``, ``dernier``); the scorer is
given one that keeps every letter and digit of any script, lower-cased.

Example:
    >>> rouge_l_f_measure("L'été dernier.", "l'été dernier")
    1.0
"""

from __future__ import annotations

import re

from rouge_score import rouge_scorer, tokenizers

# Letters and digits of any script: a word character that is not "_".
_WORD = re.compile(r"[^\W_]+")


class LettersAndDigitsTokenizer(tokenizers.Tokenizer):
    """Words of letters and digits of any script, lower-cased.

    Example:
        >>> LettersAndDigitsTokenizer().tokenize("L'été 2024")
        ['l', 'été', '2024']
    """

    def tokenize(self, text: str) -> list[str]:
        """Return the words of ``text``, lower-cased, without punctuation.

        Args:
            text: An answer's text.

        Returns:
            Its runs of letters and digits, in order.
        """
        return _WORD.findall(text.lower())


_SCORER = rouge_scorer.RougeScorer(["rougeL"], tokenizer=LettersAndDigitsTokenizer())


def rouge_l_f_measure(baseline: str, run: str) -> float:
    """Return the ROUGE-L F-measure of ``run``'s text against ``baseline``'s.

    Args:
        baseline: The baseline's answer, the reference.
        run: The run's answer.

    Returns:
        A score from 0 to 1; the library gives two empty texts 0, so a caller
        holds two equal texts the same before asking.

    Example:
        >>> rouge_l_f_measure("le chat dort", "le chat dort")
        1.0
    """
    return float(_SCORER.score(baseline, run)["rougeL"].fmeasure)
