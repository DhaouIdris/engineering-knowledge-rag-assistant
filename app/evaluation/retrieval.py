"""Page-grounded retrieval metrics for RAG experiments.

The evaluation dataset stores stable PDF page references instead of chunk IDs.
Chunk IDs change whenever chunk size or overlap changes, whereas the source
page remains stable across those experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_evaluation_dataset(path: str | Path) -> dict[str, Any]:
    """Load and validate the minimum contract of an evaluation dataset."""
    dataset_path = Path(path)
    with dataset_path.open(encoding="utf-8") as stream:
        dataset = json.load(stream)

    examples = dataset.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("The evaluation dataset must contain a non-empty 'examples' list.")

    seen_ids: set[str] = set()
    for example in examples:
        example_id = example.get("id")
        if not example_id or example_id in seen_ids:
            raise ValueError(f"Missing or duplicate example id: {example_id!r}")
        seen_ids.add(example_id)

        if not example.get("question"):
            raise ValueError(f"Example {example_id} has no question.")

        sources = example.get("relevant_sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Example {example_id} has no relevant_sources.")

        for source in sources:
            pdf_page = source.get("pdf_page_number")
            loader_page = source.get("loader_page_index")
            if not isinstance(pdf_page, int) or not isinstance(loader_page, int):
                raise ValueError(f"Example {example_id} has invalid page metadata.")
            if loader_page != pdf_page - 1:
                raise ValueError(
                    f"Example {example_id} has inconsistent one-based and zero-based pages."
                )

    return dataset


def source_basename(value: object) -> str:
    """Return a comparable filename for POSIX or Windows-style source paths."""
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def relevant_page_indices(example: Mapping[str, Any]) -> set[int]:
    """Return the zero-based PDF pages marked relevant for one question."""
    return {
        int(source["loader_page_index"])
        for source in example.get("relevant_sources", [])
    }


def evaluate_ranked_documents(
    documents: Sequence[Any],
    *,
    relevant_pages: set[int],
    expected_filename: str,
    k: int,
) -> dict[str, Any]:
    """Evaluate one ranked result list against page-level ground truth.

    A chunk is relevant when both its source filename and zero-based page index
    match the ground truth. Precision counts relevant chunks. Recall measures
    coverage of unique relevant pages, which avoids duplicate overlapping chunks
    artificially increasing recall.
    """
    if k <= 0:
        raise ValueError("k must be strictly positive.")
    if not relevant_pages:
        raise ValueError("relevant_pages must not be empty.")

    ranked = list(documents[:k])
    relevant_ranks: list[int] = []
    matched_pages: set[int] = set()
    retrieved: list[dict[str, Any]] = []

    for rank, document in enumerate(ranked, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        page = metadata.get("page")
        filename = source_basename(metadata.get("source"))
        is_relevant = filename == expected_filename and page in relevant_pages

        if is_relevant:
            relevant_ranks.append(rank)
            matched_pages.add(int(page))

        retrieved.append(
            {
                "rank": rank,
                "source": filename,
                "loader_page_index": page,
                "pdf_page_number": page + 1 if isinstance(page, int) else None,
                "relevant": is_relevant,
            }
        )

    retrieved_count = len(ranked)
    hit_at_k = 1.0 if relevant_ranks else 0.0
    precision_at_k = len(relevant_ranks) / retrieved_count if retrieved_count else 0.0
    recall_at_k = len(matched_pages) / len(relevant_pages)
    reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "matched_loader_page_indices": sorted(matched_pages),
        "retrieved": retrieved,
    }


def mean_metric(rows: Iterable[Mapping[str, Any]], metric: str) -> float:
    """Return a macro average over questions."""
    values = [float(row[metric]) for row in rows]
    return sum(values) / len(values) if values else 0.0

