"""Page-grounded retrieval metrics for RAG experiments.

The evaluation dataset stores stable PDF page references instead of chunk IDs.
Chunk IDs change whenever chunk size or overlap changes, whereas the source
page remains stable across those experiments.
"""

from __future__ import annotations

import json
import re
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


def evaluate_ranked_documents_by_locations(
    documents: Sequence[Any],
    *,
    relevant_locations: set[tuple[str, int]],
    k: int,
    page_tolerance: int = 1,
    evidence_texts: Sequence[tuple[str, str]] | None = None,
    evidence_ngram_size: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval against ``(source filename, zero-based page)`` pairs.

    Chunk precision intentionally counts duplicate relevant chunks because each
    chunk consumes context space. Unique-page precision and redundancy expose
    the effect of overlapping chunks separately.
    """
    if k <= 0:
        raise ValueError("k must be strictly positive.")
    if not relevant_locations:
        raise ValueError("relevant_locations must not be empty.")
    if page_tolerance < 0:
        raise ValueError("page_tolerance must be nonnegative.")
    if evidence_ngram_size <= 0:
        raise ValueError("evidence_ngram_size must be strictly positive.")

    def text_ngrams(text: str) -> set[tuple[str, ...]]:
        tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower().replace(",", ""))
        return {
            tuple(tokens[index : index + evidence_ngram_size])
            for index in range(len(tokens) - evidence_ngram_size + 1)
        }

    evidence_ngrams: set[tuple[str, tuple[str, ...]]] = set()
    for filename, text in evidence_texts or []:
        evidence_ngrams.update(
            (source_basename(filename), ngram) for ngram in text_ngrams(text)
        )

    ranked = list(documents[:k])
    relevant_ranks: list[int] = []
    relaxed_relevant_ranks: list[int] = []
    relevant_document_ranks: list[int] = []
    evidence_ranks: list[int] = []
    relevant_filenames = {filename for filename, _ in relevant_locations}
    pages_by_filename: dict[str, set[int]] = {}
    for filename, page in relevant_locations:
        pages_by_filename.setdefault(filename, set()).add(page)
    matched_locations: set[tuple[str, int]] = set()
    relaxed_matched_locations: set[tuple[str, int]] = set()
    matched_evidence_ngrams: set[tuple[str, tuple[str, ...]]] = set()
    retrieved_locations: set[tuple[str, int]] = set()
    retrieved: list[dict[str, Any]] = []

    for rank, document in enumerate(ranked, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        page = metadata.get("page")
        filename = source_basename(metadata.get("source"))
        location = (filename, page) if isinstance(page, int) else None
        is_relevant = location in relevant_locations if location else False
        is_relevant_document = filename in relevant_filenames
        relaxed_matches: set[tuple[str, int]] = set()
        page_distance = None
        if isinstance(page, int) and is_relevant_document:
            page_distance = min(abs(page - expected) for expected in pages_by_filename[filename])
            relaxed_matches = {
                (filename, expected)
                for expected in pages_by_filename[filename]
                if abs(page - expected) <= page_tolerance
            }
        content = str(getattr(document, "page_content", "") or "")
        document_evidence_ngrams = {
            (filename, ngram) for ngram in text_ngrams(content)
        }
        evidence_matches = document_evidence_ngrams & evidence_ngrams

        if location:
            retrieved_locations.add(location)
        if is_relevant_document:
            relevant_document_ranks.append(rank)
        if is_relevant:
            relevant_ranks.append(rank)
            matched_locations.add(location)
        if relaxed_matches:
            relaxed_relevant_ranks.append(rank)
            relaxed_matched_locations.update(relaxed_matches)
        if evidence_matches:
            evidence_ranks.append(rank)
            matched_evidence_ngrams.update(evidence_matches)

        retrieved.append(
            {
                "rank": rank,
                "source": filename,
                "loader_page_index": page,
                "pdf_page_number": page + 1 if isinstance(page, int) else None,
                "relevant": is_relevant,
                "relaxed_relevant": bool(relaxed_matches),
                "relevant_document": is_relevant_document,
                "evidence_text_match": bool(evidence_matches),
                "distance_to_nearest_relevant_page": page_distance,
                "text_preview": " ".join(
                    str(getattr(document, "page_content", "") or "").split()
                )[:500],
            }
        )

    retrieved_count = len(ranked)
    unique_location_count = len(retrieved_locations)
    matched_sorted = sorted(matched_locations)
    return {
        "hit_at_k": 1.0 if relevant_ranks else 0.0,
        "precision_at_k": len(relevant_ranks) / retrieved_count if retrieved_count else 0.0,
        "unique_page_precision_at_k": (
            len(matched_locations) / unique_location_count if unique_location_count else 0.0
        ),
        "recall_at_k": len(matched_locations) / len(relevant_locations),
        "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "relaxed_hit_at_k": 1.0 if relaxed_relevant_ranks else 0.0,
        "relaxed_precision_at_k": (
            len(relaxed_relevant_ranks) / retrieved_count if retrieved_count else 0.0
        ),
        "relaxed_recall_at_k": len(relaxed_matched_locations) / len(relevant_locations),
        "relaxed_reciprocal_rank": (
            1.0 / relaxed_relevant_ranks[0] if relaxed_relevant_ranks else 0.0
        ),
        "page_tolerance": page_tolerance,
        "evidence_hit_at_k": 1.0 if evidence_ranks else 0.0,
        "evidence_coverage_at_k": (
            len(matched_evidence_ngrams) / len(evidence_ngrams) if evidence_ngrams else 0.0
        ),
        "evidence_reciprocal_rank": 1.0 / evidence_ranks[0] if evidence_ranks else 0.0,
        "evidence_ngram_size": evidence_ngram_size,
        "document_hit_at_k": 1.0 if relevant_document_ranks else 0.0,
        "document_reciprocal_rank": (
            1.0 / relevant_document_ranks[0] if relevant_document_ranks else 0.0
        ),
        "redundancy_rate": (
            1.0 - unique_location_count / retrieved_count if retrieved_count else 0.0
        ),
        "unique_location_count": unique_location_count,
        "matched_locations": [
            {"source": filename, "loader_page_index": page}
            for filename, page in matched_sorted
        ],
        "expected_locations": [
            {"source": filename, "loader_page_index": page}
            for filename, page in sorted(relevant_locations)
        ],
        "retrieved": retrieved,
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
    if not relevant_pages:
        raise ValueError("relevant_pages must not be empty.")
    result = evaluate_ranked_documents_by_locations(
        documents,
        relevant_locations={(expected_filename, page) for page in relevant_pages},
        k=k,
    )
    result["matched_loader_page_indices"] = sorted(
        location["loader_page_index"] for location in result["matched_locations"]
    )
    return result


def mean_metric(rows: Iterable[Mapping[str, Any]], metric: str) -> float:
    """Return a macro average over questions."""
    values = [float(row[metric]) for row in rows]
    return sum(values) / len(values) if values else 0.0
