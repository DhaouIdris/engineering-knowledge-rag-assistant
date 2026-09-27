"""Small, auditable generation workflow for document-grounded FinanceBench QA.

The deterministic route deliberately covers only explicitly defined net working
capital questions. Other arithmetic requires a separately validated tool schema.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Callable, Sequence

from app.evaluation.generation import CITATION_PATTERN, REFUSAL_PATTERN, build_grounded_prompt


_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_VALUE = r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?"
_LABELS = ("Total current assets", "Total current liabilities")


def _table_value(sources: Sequence[dict[str, Any]], label: str, year: str):
    """First column only, and only where the table identifies its first year."""
    candidates: list[tuple[Decimal, str, tuple[str, Any]]] = []
    dated_pages = {
        (source.get("source"), source.get("loader_page_index"))
        for source in sources
        if re.search(
            rf"(?<!\d){re.escape(year)}\s+(?:19|20)\d{{2}}(?!\d)",
            str(source.get("content") or ""),
        )
    }
    for source in sources:
        content = str(source.get("content") or "")
        # Financial statements put the latest year in the first numeric column.
        location = (source.get("source"), source.get("loader_page_index"))
        if location not in dated_pages:
            continue
        match = re.search(
            rf"(?im)^\s*{re.escape(label)}\s+({_VALUE})(?=\s|$)", content
        )
        if match:
            raw = match.group(1).replace("$", "").replace(",", "").strip()
            negative = raw.startswith("(") and raw.endswith(")")
            value = Decimal(raw.strip("()")) * (-1 if negative else 1)
            candidates.append((value, source["label"], location))
    if not candidates or len({value for value, _, _ in candidates}) != 1:
        return None
    return candidates[0]


def calculate_working_capital(question: str, sources: Sequence[dict[str, Any]]):
    """Return a cited answer when both line items are unambiguous, else None.

    Never uses benchmark labels, reference answers or annotated evidence pages.
    """
    if not re.search(r"\bnet working capital\b", question, re.IGNORECASE):
        return None
    if not re.search(r"current assets\s+(?:less|minus|-)\s+(?:total )?current liabilities", question, re.IGNORECASE):
        return None
    years = set(_YEAR.findall(question))
    if len(years) != 1:
        return None
    year = years.pop()
    assets = _table_value(sources, _LABELS[0], year)
    liabilities = _table_value(sources, _LABELS[1], year)
    if assets is None or liabilities is None or assets[2] != liabilities[2]:
        return None
    value = assets[0] - liabilities[0]
    amount = f"{value:,.0f}" if value == value.to_integral_value() else f"{value:,.2f}"
    unit = " million" if re.search(r"\bmillions?\b", question, re.IGNORECASE) else ""
    answer = (
        f"FY{year} net working capital is ${amount}{unit} "
        f"(${assets[0]:,.0f} [{assets[1]}] − ${liabilities[0]:,.0f} [{liabilities[1]}])."
    )
    return {"answer": answer, "operands": {"assets": str(assets[0]), "liabilities": str(liabilities[0])}}


def check_answer(
    answer: str, sources: Sequence[dict[str, Any]], *,
    computed: bool = False, question: str = "",
) -> list[str]:
    """Conservative syntax/provenance guard; NOT a semantic entailment judge."""
    if REFUSAL_PATTERN.match(answer):
        return []
    issues: list[str] = []
    citations = [int(index) for index in CITATION_PATTERN.findall(answer)]
    if not citations or any(index < 1 or index > len(sources) for index in citations):
        issues.append("missing_or_invalid_citations")
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    for sentence in sentences:
        if len(sentence.split()) >= 3 and not CITATION_PATTERN.search(sentence):
            issues.append("uncited_sentence")
            break
    if computed:
        return list(dict.fromkeys(issues))

    # A number in a factual answer must be traceable to the cited source(s).
    # Ignore dates that merely repeat the question in the answer; unsupported
    # numerical claims are still flagged, without pretending to prove prose.
    for sentence in sentences:
        cited = [int(index) for index in CITATION_PATTERN.findall(sentence)]
        if not cited or any(index < 1 or index > len(sources) for index in cited):
            continue
        source_text = " ".join(str(sources[index - 1].get("content") or "") for index in cited)
        numbers = re.findall(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?", CITATION_PATTERN.sub("", sentence))
        for number in numbers:
            # Repeated question numbers (e.g. the report date) are user-provided,
            # not claims newly inferred from a source.
            if re.search(rf"(?<![\d,]){re.escape(number)}(?![\d,])", question):
                continue
            if not re.search(rf"(?<![\d,]){re.escape(number)}(?![\d,])", source_text):
                issues.append("unsupported_number")
                break
    return list(dict.fromkeys(issues))


def grounded_answer(
    question: str, context: str, sources: Sequence[dict[str, Any]], invoke: Callable[[str], str]
) -> dict[str, Any]:
    """Route a supported calculation; otherwise generate, check, and abstain."""
    calculation = calculate_working_capital(question, sources)
    if calculation is not None:
        candidate = calculation["answer"]
        route = "deterministic_working_capital"
    else:
        candidate = invoke(build_grounded_prompt(question, context))
        route = "llm"
    issues = check_answer(candidate, sources, computed=calculation is not None, question=question)
    answer = "INSUFFICIENT_CONTEXT: answer failed citation/provenance checks." if issues else candidate
    return {"answer": answer, "raw_answer": candidate, "route": route, "critic_issues": issues,
            "calculation": calculation["operands"] if calculation else None}
