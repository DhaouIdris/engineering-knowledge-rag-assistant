"""Rank fusion for dense and lexical retrieval results."""

from __future__ import annotations

from typing import Any, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]], k: int, constant: int = 60
) -> list[Any]:
    """Merge chunk rankings without comparing incompatible similarity scores."""
    if k <= 0 or constant < 0:
        raise ValueError("k must be positive and constant must be nonnegative.")
    scores: dict[tuple[str, str, str], float] = {}
    documents: dict[tuple[str, str, str], Any] = {}
    for ranking in rankings:
        seen: set[tuple[str, str, str]] = set()
        for rank, document in enumerate(ranking, start=1):
            key = (
                str(document.metadata.get("source", "")),
                str(document.metadata.get("page", "")),
                document.page_content,
            )
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
    return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])[:k]]
