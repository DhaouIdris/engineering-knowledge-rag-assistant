"""Small in-memory BM25 baseline for FinanceBench retrieval experiments."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Sequence


def tokenize(text: str) -> list[str]:
    """Preserve exact years and numbers in financial documents and questions."""
    lowered = text.lower()
    lowered = re.sub(r"(?<=\d),(?=\d{3}\b)", "", lowered)
    lowered = re.sub(r"\bfy(?=\d{4}\b)", "", lowered)
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", lowered)


class BM25Index:
    """Rank chunks by lexical matches; no model download or embedding needed."""

    def __init__(self, documents: Sequence[Any]) -> None:
        self.documents = list(documents)
        lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            tokens = tokenize(document.page_content)
            lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                postings[term].append((index, count))
        self.lengths = lengths
        self.avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        self.postings = postings

    def search(self, question: str, k: int) -> list[Any]:
        if k <= 0:
            raise ValueError("k must be strictly positive.")
        size = len(self.documents)
        if not size:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(question)):
            matches = self.postings.get(term, [])
            if not matches:
                continue
            inverse_frequency = math.log(1 + (size - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, frequency in matches:
                normalizer = 1.2 * (0.25 + 0.75 * self.lengths[index] / self.avg_length)
                scores[index] += inverse_frequency * (frequency * 2.2) / (frequency + normalizer)

        ranked = sorted(scores, key=lambda index: (-scores[index], index))
        return [self.documents[index] for index in ranked[:k]]
