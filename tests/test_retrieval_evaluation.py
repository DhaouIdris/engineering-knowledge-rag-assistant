from dataclasses import dataclass

import pytest

from app.evaluation.retrieval import (
    evaluate_ranked_documents,
    evaluate_ranked_documents_by_locations,
    load_evaluation_dataset,
    source_basename,
)


@dataclass
class FakeDocument:
    metadata: dict
    page_content: str = ""


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
    assert result["document_hit_at_k"] == 1.0
    assert result["document_reciprocal_rank"] == 1.0
    assert result["retrieved"][0]["distance_to_nearest_relevant_page"] == 8


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
    assert result["redundancy_rate"] == pytest.approx(1 / 3)


def test_multi_document_ground_truth_and_unique_page_precision():
    result = evaluate_ranked_documents_by_locations(
        [document(2, "/data/a.pdf"), document(2, "/data/a.pdf"), document(7, "/data/b.pdf")],
        relevant_locations={("a.pdf", 2), ("b.pdf", 7)},
        k=3,
    )

    assert result["precision_at_k"] == 1.0
    assert result["unique_page_precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["redundancy_rate"] == pytest.approx(1 / 3)
    assert result["unique_location_count"] == 2


def test_diagnostics_include_text_and_separate_document_from_page_hit():
    retrieved = FakeDocument(
        metadata={"page": 7, "source": "/data/manual.pdf"},
        page_content="  A financial   table with revenue.  ",
    )
    result = evaluate_ranked_documents_by_locations(
        [retrieved], relevant_locations={("manual.pdf", 10)}, k=1
    )

    assert result["hit_at_k"] == 0.0
    assert result["document_hit_at_k"] == 1.0
    assert result["document_reciprocal_rank"] == 1.0
    assert result["retrieved"][0]["distance_to_nearest_relevant_page"] == 3
    assert result["retrieved"][0]["text_preview"] == "A financial table with revenue."


def test_relaxed_metrics_accept_only_nearby_pages_from_the_same_document():
    result = evaluate_ranked_documents_by_locations(
        [document(11), document(10, "/data/another.pdf")],
        relevant_locations={("manual.pdf", 10)},
        k=2,
        page_tolerance=1,
    )

    assert result["hit_at_k"] == 0.0
    assert result["relaxed_hit_at_k"] == 1.0
    assert result["relaxed_precision_at_k"] == 0.5
    assert result["relaxed_recall_at_k"] == 1.0
    assert result["relaxed_reciprocal_rank"] == 1.0
    assert result["retrieved"][0]["relaxed_relevant"] is True
    assert result["retrieved"][1]["relaxed_relevant"] is False


def test_evidence_metrics_measure_normalized_ngram_coverage():
    first = FakeDocument(
        metadata={"page": 4, "source": "/data/manual.pdf"},
        page_content="Unrelated introductory material.",
    )
    second = FakeDocument(
        metadata={"page": 7, "source": "/data/manual.pdf"},
        page_content="Revenue increased to $1,577 million during FY2018.",
    )
    result = evaluate_ranked_documents_by_locations(
        [first, second],
        relevant_locations={("manual.pdf", 10)},
        k=2,
        evidence_texts=[
            ("manual.pdf", "Revenue increased to 1,577 million during FY2018 due to demand.")
        ],
        evidence_ngram_size=5,
    )

    assert result["hit_at_k"] == 0.0
    assert result["evidence_hit_at_k"] == 1.0
    assert result["evidence_coverage_at_k"] > 0.0
    assert result["evidence_reciprocal_rank"] == 0.5
    assert result["retrieved"][1]["evidence_text_match"] is True


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
