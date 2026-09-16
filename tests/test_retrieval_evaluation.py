from dataclasses import dataclass

import pytest

from app.evaluation.retrieval import (
    evaluate_ranked_documents,
    load_evaluation_dataset,
    source_basename,
)


@dataclass
class FakeDocument:
    metadata: dict


def document(page: int, source: str = "/data/manual.pdf") -> FakeDocument:
    return FakeDocument(metadata={"page": page, "source": source})


def test_ranked_metrics_use_source_and_page_ground_truth():
    result = evaluate_ranked_documents(
        [document(2), document(10), document(11)],
        relevant_pages={10, 11},
        expected_filename="manual.pdf",
        k=3,
    )

    assert result["hit_at_k"] == 1.0
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank"] == 0.5


def test_duplicate_chunks_do_not_inflate_page_recall():
    result = evaluate_ranked_documents(
        [document(10), document(10), document(4)],
        relevant_pages={10, 11},
        expected_filename="manual.pdf",
        k=3,
    )

    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == 0.5
    assert result["matched_loader_page_indices"] == [10]


def test_same_page_from_another_document_is_not_relevant():
    result = evaluate_ranked_documents(
        [document(10, "/data/another.pdf")],
        relevant_pages={10},
        expected_filename="manual.pdf",
        k=1,
    )

    assert result["hit_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_source_basename_supports_windows_paths():
    assert source_basename(r"C:\\documents\\manual.pdf") == "manual.pdf"


def test_repository_dataset_contract():
    dataset = load_evaluation_dataset(
        "evaluations/datasets/automobile_engineering_v1.json"
    )
    assert dataset["dataset_name"] == "automobile_engineering_rag_eval_v1"
    assert len(dataset["examples"]) == 10

