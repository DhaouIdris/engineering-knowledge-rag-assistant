import json

import pytest

from app.evaluation.financebench import load_financebench_dataset


def write_questions(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def financebench_row(example_id="financebench_id_1", doc_name="ACME_2024_10K"):
    return {
        "financebench_id": example_id,
        "question": "What was ACME revenue?",
        "answer": "$10 million",
        "justification": "Reported revenue.",
        "question_type": "metrics-generated",
        "question_reasoning": "Information extraction",
        "evidence": [
            {"doc_name": doc_name, "evidence_page_num": 0, "evidence_text": "Revenue table"},
            {"doc_name": doc_name, "evidence_page_num": 0},
            {"doc_name": doc_name, "evidence_page_num": 2},
        ],
    }


def test_load_financebench_resolves_pdfs_and_deduplicates_evidence(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "ACME_2024_10K.pdf").touch()
    questions = tmp_path / "questions.jsonl"
    write_questions(questions, [financebench_row()])

    dataset = load_financebench_dataset(questions, pdf_dir)
    example = dataset["examples"][0]

    assert example["id"] == "financebench_id_1"
    assert example["relevant_locations"] == {
        ("ACME_2024_10K.pdf", 0),
        ("ACME_2024_10K.pdf", 2),
    }
    assert example["evidence"][0]["evidence_text"] == "Revenue table"


def test_load_financebench_reports_missing_pdf(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "OTHER.pdf").touch()
    questions = tmp_path / "questions.jsonl"
    write_questions(questions, [financebench_row()])

    with pytest.raises(ValueError, match="was not found"):
        load_financebench_dataset(questions, pdf_dir)


def test_load_financebench_limit(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "ACME_2024_10K.pdf").touch()
    questions = tmp_path / "questions.jsonl"
    write_questions(
        questions,
        [financebench_row("financebench_id_1"), financebench_row("financebench_id_2")],
    )

    dataset = load_financebench_dataset(questions, pdf_dir, limit=1)
    assert len(dataset["examples"]) == 1
