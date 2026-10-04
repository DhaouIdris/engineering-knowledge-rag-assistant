import pytest

pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_community")

from scripts.evaluate_financebench_generation import (  # noqa: E402
    adjacent_page_chunks,
    indexed_page_chunks,
    append_checkpoint,
    load_checkpoint,
)


def test_generation_checkpoint_round_trip_and_signature_validation(tmp_path):
    checkpoint = tmp_path / "answers.jsonl"
    append_checkpoint(
        checkpoint, {"run_signature": "run-a", "id": "question-1", "answer": "A"}
    )
    append_checkpoint(
        checkpoint, {"run_signature": "run-a", "id": "question-2", "answer": "B"}
    )

    completed = load_checkpoint(checkpoint, "run-a")

    assert set(completed) == {"question-1", "question-2"}
    assert completed["question-2"]["answer"] == "B"
    with pytest.raises(ValueError, match="configuration differs"):
        load_checkpoint(checkpoint, "run-b")


def test_oracle_page_context_uses_only_indexed_chunks_on_annotated_pages():
    from langchain_core.documents import Document

    class Docstore:
        def search(self, document_id):
            return {
                "other": Document(page_content="Wrong page", metadata={"source": "report.pdf", "page": 7}),
                "gold": Document(page_content="Correct page", metadata={"source": "report.pdf", "page": 3}),
            }[document_id]

    class Store:
        docstore = Docstore()
        index_to_docstore_id = {0: "other", 1: "gold"}

    selected = indexed_page_chunks(Store(), {("report.pdf", 3)}, limit=2)
    assert [document.page_content for document in selected] == ["Correct page"]


def test_page_neighbors_include_header_and_row_without_unrelated_pages():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="unrelated", metadata={"source": "report.pdf", "page": 2}),
        Document(page_content="FY2024 FY2023", metadata={"source": "report.pdf", "page": 3}),
        Document(page_content="Total stores 969 982", metadata={"source": "report.pdf", "page": 3}),
        Document(page_content="other table", metadata={"source": "report.pdf", "page": 3}),
    ]

    class Store:
        index_to_docstore_id = dict(enumerate(range(len(docs))))
        docstore = type("Docstore", (), {"search": lambda self, i: docs[i]})()

    selected = adjacent_page_chunks(Store(), docs[2], limit=3)
    assert [doc.page_content for doc in selected] == [
        "FY2024 FY2023", "Total stores 969 982", "other table"
    ]
    assert [doc.page_content for doc in adjacent_page_chunks(Store(), docs[2], limit=1)] == [
        "Total stores 969 982"
    ]
