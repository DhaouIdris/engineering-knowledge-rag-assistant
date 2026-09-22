from dataclasses import dataclass, field

import pytest

from app.evaluation.generation import (
    citation_metrics,
    evaluate_answer,
    exact_match,
    expand_with_same_page_chunks,
    format_context,
    numeric_recall,
    stratified_sample,
    token_f1,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeDocstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, document_id):
        return self.documents[document_id]


@dataclass
class FakeStore:
    documents: dict

    def __post_init__(self):
        self.docstore = FakeDocstore(self.documents)
        self.index_to_docstore_id = dict(enumerate(self.documents))


def test_stratified_sample_is_balanced_and_reproducible():
    examples = [
        {"id": f"{question_type}-{index}", "question_type": question_type}
        for question_type in ("domain", "metrics", "novel")
        for index in range(5)
    ]

    first = stratified_sample(examples, samples_per_type=2, seed=42)
    second = stratified_sample(list(reversed(examples)), samples_per_type=2, seed=42)

    assert [example["id"] for example in first] == [
        example["id"] for example in second
    ]
    assert len(first) == 6
    assert {question_type: sum(e["question_type"] == question_type for e in first)
            for question_type in ("domain", "metrics", "novel")} == {
        "domain": 2,
        "metrics": 2,
        "novel": 2,
    }


def test_format_context_creates_stable_source_labels():
    context, sources = format_context(
        [
            FakeDocument("First passage", {"source": "/docs/report.pdf", "page": 4}),
            FakeDocument("Second passage", {"source": "/docs/report.pdf", "page": 8}),
        ]
    )

    assert "[S1] report.pdf, PDF page 5" in context
    assert "[S2] report.pdf, PDF page 9" in context
    assert sources[0]["loader_page_index"] == 4


def test_context_expansion_adds_companion_chunks_from_top_page_only():
    first = FakeDocument("assets", {"source": "report.pdf", "page": 4})
    companion = FakeDocument("liabilities", {"source": "report.pdf", "page": 4})
    unrelated = FakeDocument("notes", {"source": "report.pdf", "page": 8})
    store = FakeStore({"a": first, "b": companion, "c": unrelated})

    expanded = expand_with_same_page_chunks(
        [first], store, top_pages=1, max_documents=5
    )

    assert [document.page_content for document in expanded] == [
        "assets",
        "liabilities",
    ]


def test_context_expansion_prioritizes_the_highest_ranked_page():
    top = FakeDocument("top result", {"source": "report.pdf", "page": 9})
    lower = FakeDocument("lower result", {"source": "report.pdf", "page": 1})
    top_companion = FakeDocument("needed row", {"source": "report.pdf", "page": 9})
    lower_companion = FakeDocument("old note", {"source": "report.pdf", "page": 1})
    store = FakeStore(
        {"a": lower, "b": lower_companion, "c": top, "d": top_companion}
    )

    expanded = expand_with_same_page_chunks(
        [top, lower], store, top_pages=2, max_documents=3
    )

    assert [document.page_content for document in expanded] == [
        "top result",
        "lower result",
        "needed row",
    ]


def test_answer_similarity_and_numeric_metrics_are_deterministic():
    assert exact_match("$1,577.00 [S1]", "$1577.00") == 1.0
    assert token_f1("Revenue increased strongly", "Revenue increased") == pytest.approx(
        0.8
    )
    assert numeric_recall("The result is $1,577 [S1].", "$1577.00") == 1.0
    assert numeric_recall("No numeric reference", "Yes") is None


def test_citation_metrics_validate_labels_and_ground_truth_pages():
    sources = [
        {"source": "report.pdf", "loader_page_index": 4},
        {"source": "report.pdf", "loader_page_index": 8},
    ]
    metrics = citation_metrics(
        "Revenue was $10 million [S1]. Another unsupported statement. [S99]",
        sources,
        {("report.pdf", 4)},
    )

    assert metrics["citation_presence"] == 1.0
    assert metrics["citation_validity"] == 0.5
    assert metrics["citation_ground_truth_precision"] == 1.0
    assert metrics["citation_ground_truth_recall"] == 1.0
    assert metrics["citation_ground_truth_hit"] == 1.0


def test_citation_evidence_hit_accepts_same_evidence_on_an_alternative_page():
    sources = [
        {
            "source": "report.pdf",
            "loader_page_index": 8,
            "content": "The company generated 13.2 billion in cash proceeds from the offering.",
        }
    ]
    metrics = citation_metrics(
        "Cash proceeds were $13.2 billion [S1].",
        sources,
        {("report.pdf", 4)},
        evidence_texts=[
            (
                "report.pdf",
                "The company secured 13.2 billion in cash proceeds from the offering.",
            )
        ],
    )

    assert metrics["citation_ground_truth_hit"] == 0.0
    assert metrics["citation_evidence_hit"] == 1.0


def test_refusal_is_appropriate_when_retrieved_context_has_no_evidence():
    metrics = evaluate_answer(
        "INSUFFICIENT_CONTEXT The required balance sheet is missing.",
        "42",
        [],
        {("report.pdf", 4)},
        context_evidence_hit=False,
    )

    assert metrics["refusal"] == 1.0
    assert metrics["evidence_action_alignment"] == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "INSUFFICIENT_CONTEXT The required balance sheet is missing.",
        "INSUFFICIENT CONTEXT\nThe required balance sheet is missing.",
        "  insufficient context: the required balance sheet is missing.",
    ],
)
def test_refusal_detection_accepts_common_model_variants(answer):
    metrics = evaluate_answer(
        answer,
        "42",
        [],
        {("report.pdf", 4)},
        context_evidence_hit=False,
    )

    assert metrics["refusal"] == 1.0
    assert metrics["evidence_action_alignment"] == 1.0
