"""FinanceBench JSONL adapter with PDF and page-level validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _normalise_document_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(value).stem.lower())


def build_pdf_lookup(pdf_dir: str | Path) -> dict[str, str]:
    """Map normalised FinanceBench document names to actual PDF filenames."""
    directory = Path(pdf_dir)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found in {directory}.")

    lookup: dict[str, str] = {}
    for pdf in pdfs:
        key = _normalise_document_name(pdf.name)
        if key in lookup:
            raise ValueError(
                f"Ambiguous PDF names after normalisation: {lookup[key]!r} and {pdf.name!r}."
            )
        lookup[key] = pdf.name
    return lookup


def load_financebench_dataset(
    questions_path: str | Path,
    pdf_dir: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Load official FinanceBench questions and resolve evidence to local PDFs."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be strictly positive.")

    lookup = build_pdf_lookup(pdf_dir)
    rows: list[dict[str, Any]] = []
    with Path(questions_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError("FinanceBench question file is empty.")

    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        example_id = row.get("financebench_id")
        if not example_id or example_id in seen_ids:
            raise ValueError(f"Missing or duplicate financebench_id: {example_id!r}")
        seen_ids.add(example_id)
        if not row.get("question"):
            raise ValueError(f"Question {example_id} has no question text.")

        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Question {example_id} has no evidence.")

        locations: set[tuple[str, int]] = set()
        for item in evidence:
            document_name = item.get("doc_name")
            page = item.get("evidence_page_num")
            if not isinstance(document_name, str) or not isinstance(page, int) or page < 0:
                raise ValueError(f"Question {example_id} has invalid evidence metadata.")
            key = _normalise_document_name(document_name)
            if key not in lookup:
                raise ValueError(
                    f"Question {example_id} references {document_name!r}, but its PDF "
                    f"was not found in {Path(pdf_dir)}."
                )
            locations.add((lookup[key], page))

        examples.append(
            {
                "id": example_id,
                "question": row["question"],
                "answer": row.get("answer"),
                "justification": row.get("justification"),
                "question_type": row.get("question_type"),
                "question_reasoning": row.get("question_reasoning"),
                "relevant_locations": locations,
            }
        )

    return {"dataset_name": "financebench_open_source", "examples": examples}
