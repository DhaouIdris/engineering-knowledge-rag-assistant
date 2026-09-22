"""Deterministic metrics and formatting for grounded answer evaluation."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


CITATION_PATTERN = re.compile(r"\[S(\d+)\]", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
REFUSAL_MARKER = "INSUFFICIENT_CONTEXT"
REFUSAL_PATTERN = re.compile(r"^\s*INSUFFICIENT(?:_|\s+)CONTEXT\b", re.IGNORECASE)


def stratified_sample(
    examples: Sequence[dict[str, Any]], samples_per_type: int, seed: int
) -> list[dict[str, Any]]:
    """Select a stable sample from every FinanceBench question type."""
    if samples_per_type <= 0:
        raise ValueError("samples_per_type must be strictly positive.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[str(example.get("question_type") or "unknown")].append(example)

    selected: list[dict[str, Any]] = []
    for question_type in sorted(groups):
        group = groups[question_type]
        if len(group) < samples_per_type:
            raise ValueError(
                f"Question type {question_type!r} contains only {len(group)} examples."
            )
        ranked = sorted(
            group,
            key=lambda example: hashlib.sha256(
                f"{seed}:{example['id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:samples_per_type])
    return sorted(selected, key=lambda example: example["id"])


def source_basename(value: Any) -> str:
    return Path(str(value or "")).name


def format_context(documents: Sequence[Any]) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        source = source_basename(document.metadata.get("source"))
        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        label = f"S{index}"
        page_text = f", PDF page {page_number}" if page_number is not None else ""
        content = str(document.page_content or "").strip()
        blocks.append(f"[{label}] {source}{page_text}\n{content}")
        sources.append(
            {
                "label": label,
                "source": source,
                "loader_page_index": page,
                "pdf_page_number": page_number,
                "content": content,
            }
        )
    return "\n\n".join(blocks), sources


def expand_with_same_page_chunks(
    documents: Sequence[Any],
    store: Any,
    *,
    top_pages: int = 3,
    max_documents: int = 20,
) -> list[Any]:
    """Add companion chunks from the pages of the highest-ranked results.

    Financial tables are frequently split into several chunks even though all
    rows belong to one PDF page. Retrieval ranking remains based on the original
    results; this function only expands the context given to the generator.
    """
    if top_pages <= 0 or max_documents <= 0:
        return list(documents)[:max_documents]

    selected = list(documents)[:max_documents]
    selected_keys = {
        (
            source_basename(document.metadata.get("source")),
            document.metadata.get("page"),
            str(document.page_content or ""),
        )
        for document in selected
    }
    target_pages: list[tuple[str, int]] = []
    for document in documents:
        page = document.metadata.get("page")
        location = (source_basename(document.metadata.get("source")), page)
        if isinstance(page, int) and location not in target_pages:
            target_pages.append(location)
        if len(target_pages) >= top_pages:
            break

    docstore = getattr(store, "docstore", None)
    index_to_id = getattr(store, "index_to_docstore_id", {})
    if docstore is None:
        return selected

    candidates_by_location: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for document_id in index_to_id.values():
        candidate = docstore.search(document_id)
        if not hasattr(candidate, "metadata"):
            continue
        location = (
            source_basename(candidate.metadata.get("source")),
            candidate.metadata.get("page"),
        )
        if location in target_pages:
            candidates_by_location[location].append(candidate)

    # Preserve retrieval priority: exhaust companions from the highest-ranked
    # page before lower-ranked pages can consume the context budget.
    for location in target_pages:
        for candidate in candidates_by_location[location]:
            key = (*location, str(candidate.page_content or ""))
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) >= max_documents:
                return selected
    return selected


def build_grounded_prompt(question: str, context: str) -> str:
    return f"""You are a financial document question-answering assistant.

Use only the supplied sources. Do not rely on outside knowledge.
First decide whether the sources contain every fact or number needed to answer.
A source is sufficient even when it uses different wording from the question.
If the needed inputs are present, perform the requested arithmetic yourself;
do not refuse merely because the final calculated metric is not written out.

When the evidence is sufficient, answer the question directly in the first
sentence. Answer in English and be concise, showing essential arithmetic when
required. Cite every factual statement with one or more source labels such as
[S1] or [S2]. Every non-refusal answer must contain at least one citation.
Never invent a source label. Before returning a non-refusal answer, verify that
it contains a bracketed source label.

Example of the required answer format:
Revenue was $10 million [S1].

Only when at least one required fact or number is absent, begin the answer with
exactly {REFUSAL_MARKER} and briefly state what is missing. Do not put a space
between INSUFFICIENT and CONTEXT.

Question:
{question}

Sources:
{context}

Answer:
"""


def strip_citations(text: str) -> str:
    return CITATION_PATTERN.sub(" ", text)


def answer_tokens(text: str) -> list[str]:
    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", strip_citations(text).lower())
    return TOKEN_PATTERN.findall(normalized)


def exact_match(prediction: str, reference: str) -> float:
    return float(answer_tokens(prediction) == answer_tokens(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(answer_tokens(prediction))
    expected = Counter(answer_tokens(reference))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    common = sum((predicted & expected).values())
    precision = common / sum(predicted.values())
    recall = common / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if common else 0.0


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_PATTERN.findall(strip_citations(text)):
        normalized = match.replace("$", "").replace(",", "").replace("%", "")
        try:
            values.append(float(normalized))
        except ValueError:
            continue
    return values


def _number_matches(expected: float, predicted: float) -> bool:
    tolerance = max(0.01, abs(expected) * 1e-4)
    return math.isclose(expected, predicted, rel_tol=0.0, abs_tol=tolerance)


def numeric_recall(prediction: str, reference: str) -> float | None:
    expected = extract_numbers(reference)
    if not expected:
        return None
    predicted = extract_numbers(prediction)
    matched = sum(
        any(_number_matches(reference_value, value) for value in predicted)
        for reference_value in expected
    )
    return matched / len(expected)


def _substantive_sentences(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    return [sentence for sentence in sentences if len(answer_tokens(sentence)) >= 3]


def citation_metrics(
    answer: str,
    sources: Sequence[dict[str, Any]],
    relevant_locations: set[tuple[str, int]],
    *,
    page_tolerance: int = 1,
    evidence_texts: Sequence[tuple[str, str]] | None = None,
    evidence_ngram_size: int = 5,
) -> dict[str, float]:
    if evidence_ngram_size <= 0:
        raise ValueError("evidence_ngram_size must be strictly positive.")
    cited_numbers = {int(value) for value in CITATION_PATTERN.findall(answer)}
    valid_numbers = {value for value in cited_numbers if 1 <= value <= len(sources)}
    cited_sources = [sources[value - 1] for value in sorted(valid_numbers)]

    exact_matches: set[tuple[str, int]] = set()
    relaxed_matches: set[tuple[str, int]] = set()
    for source in cited_sources:
        filename = source_basename(source.get("source"))
        page = source.get("loader_page_index")
        if not isinstance(page, int):
            continue
        location = (filename, page)
        if location in relevant_locations:
            exact_matches.add(location)
        for expected_filename, expected_page in relevant_locations:
            if filename == expected_filename and abs(page - expected_page) <= page_tolerance:
                relaxed_matches.add((expected_filename, expected_page))

    exact_cited = sum(
        (source_basename(source.get("source")), source.get("loader_page_index"))
        in relevant_locations
        for source in cited_sources
    )
    sentences = _substantive_sentences(answer)
    cited_sentences = sum(bool(CITATION_PATTERN.search(sentence)) for sentence in sentences)
    evidence_ngrams: set[tuple[str, tuple[str, ...]]] = set()
    for filename, text in evidence_texts or []:
        tokens = answer_tokens(text)
        evidence_ngrams.update(
            (source_basename(filename), tuple(tokens[index : index + evidence_ngram_size]))
            for index in range(len(tokens) - evidence_ngram_size + 1)
        )
    cited_evidence_ngrams: set[tuple[str, tuple[str, ...]]] = set()
    for source in cited_sources:
        filename = source_basename(source.get("source"))
        tokens = answer_tokens(str(source.get("content") or ""))
        source_ngrams = {
            (filename, tuple(tokens[index : index + evidence_ngram_size]))
            for index in range(len(tokens) - evidence_ngram_size + 1)
        }
        cited_evidence_ngrams.update(source_ngrams & evidence_ngrams)
    valid_count = len(valid_numbers)
    citation_count = len(cited_numbers)
    return {
        "citation_presence": float(bool(cited_numbers)),
        "citation_validity": valid_count / citation_count if citation_count else 0.0,
        "citation_sentence_coverage": (
            cited_sentences / len(sentences) if sentences else 0.0
        ),
        "citation_ground_truth_precision": (
            exact_cited / valid_count if valid_count else 0.0
        ),
        "citation_ground_truth_recall": (
            len(exact_matches) / len(relevant_locations) if relevant_locations else 0.0
        ),
        "citation_ground_truth_hit": float(bool(exact_matches)),
        "relaxed_citation_ground_truth_hit": float(bool(relaxed_matches)),
        "citation_evidence_hit": float(bool(cited_evidence_ngrams)),
        "citation_evidence_coverage": (
            len(cited_evidence_ngrams) / len(evidence_ngrams)
            if evidence_ngrams
            else 0.0
        ),
    }


def evaluate_answer(
    answer: str,
    reference: str,
    sources: Sequence[dict[str, Any]],
    relevant_locations: set[tuple[str, int]],
    *,
    context_evidence_hit: bool,
    evidence_texts: Sequence[tuple[str, str]] | None = None,
) -> dict[str, float | None]:
    refused = bool(REFUSAL_PATTERN.match(answer))
    metrics: dict[str, float | None] = {
        "exact_match": exact_match(answer, reference),
        "token_f1": token_f1(answer, reference),
        "numeric_recall": numeric_recall(answer, reference),
        "refusal": float(refused),
        "evidence_action_alignment": float(
            (context_evidence_hit and not refused) or (not context_evidence_hit and refused)
        ),
    }
    metrics.update(
        citation_metrics(
            answer,
            sources,
            relevant_locations,
            evidence_texts=evidence_texts,
        )
    )
    return metrics


def mean_available(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None
