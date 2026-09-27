"""Small, auditable generation workflow for document-grounded FinanceBench QA.

Deterministic routes handle only calculations whose operands, dates and units
can be read unambiguously from cited passages.
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Sequence

from app.evaluation.generation import CITATION_PATTERN, REFUSAL_PATTERN, build_grounded_prompt


_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_VALUE = r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?"
_LABELS = ("Total current assets", "Total current liabilities")


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def calculate_credit_capacity(question: str, sources: Sequence[dict[str, Any]]):
    """Sum distinct new revolving facilities effective on the asked date.

    Optional commitment increases and terminated facilities are not capacity.
    Refuse if any facility is duplicated with conflicting amounts.
    """
    if not (re.search(r"\btotal amount\b.*\bborrow\b", question, re.I)
            and re.search(r"\brevolving credit agreements\b", question, re.I)):
        return None
    asked_date = re.search(r"\b(?:as of|on)\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})", question, re.I)
    if not asked_date:
        return None
    facilities: dict[str, tuple[Decimal, str, tuple[Any, Any]]] = {}
    for source in sources:
        content = _normal(str(source.get("content") or ""))
        # A statement beginning "entered into a new" excludes historic,
        # terminated and merely optional increased commitments.
        for match in re.finditer(
            r"\bOn\s+([A-Za-z]+\s+\d{1,2},\s*\d{4}),?\s+.{0,100}?"
            r"entered into a new\s+\$\s*(\d[\d,]*(?:\.\d+)?)\s+"
            r"(five[- ]year|364[- ]day)\s+unsecured revolving credit agreement\b",
            content, re.I,
        ):
            if _normal(match.group(1)).casefold() != _normal(asked_date.group(1)).casefold():
                continue
            kind = re.sub(r"[- ]", "", match.group(3).casefold())
            amount = Decimal(match.group(2).replace(",", ""))
            location = (source.get("source"), source.get("loader_page_index"))
            previous = facilities.get(kind)
            if previous and (previous[0] != amount or previous[2] != location):
                return None
            facilities.setdefault(kind, (amount, source["label"], location))
    if set(facilities) != {"fiveyear", "364day"}:
        return None
    first, second = facilities["fiveyear"], facilities["364day"]
    if first[2] != second[2]:
        return None
    total = first[0] + second[0]
    return {
        "answer": (f"The two revolving facilities allow ${total:,.0f} in total "
                   f"(${first[0]:,.0f} [{first[1]}] + ${second[0]:,.0f} [{second[1]}])."),
        "operands": {"five_year": str(first[0]), "364_day": str(second[0])},
    }


def calculate_revenue_growth(question: str, sources: Sequence[dict[str, Any]]):
    """Calculate growth from the consolidated income statement's first two years."""
    if not (re.search(r"\b(?:total|net) revenue growth rate\b", question, re.I)
            and re.search(r"\bone decimal place\b", question, re.I)):
        return None
    years = sorted(set(_YEAR.findall(question)))
    if len(years) != 2 or int(years[1]) - int(years[0]) != 1:
        return None
    candidates: list[tuple[Decimal, Decimal, str]] = []
    for source in sources:
        content = _normal(str(source.get("content") or ""))
        if not re.search(r"CONSOLIDATED STATEMENTS OF OPERATIONS", content, re.I):
            continue
        header = re.search(rf"\b{years[1]}\s+{years[0]}\b", content)
        row = re.search(
            rf"\bTotal net revenue\s+\$?\s*({_VALUE})\s+\$?\s*({_VALUE})", content, re.I
        )
        if not header or not row or header.start() > row.start():
            continue
        current = Decimal(row.group(1).replace(",", ""))
        previous = Decimal(row.group(2).replace(",", ""))
        if previous <= 0:
            return None
        candidates.append((current, previous, source["label"]))
    if not candidates or len({(a, b) for a, b, _ in candidates}) != 1:
        return None
    current, previous, label = candidates[0]
    percentage = ((current / previous - 1) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return {
        "answer": (f"Total net revenue grew {percentage}% from {years[0]} to {years[1]} "
                   f"(({current:,.0f} − {previous:,.0f}) / {previous:,.0f} × 100) [{label}]."),
        "operands": {"current": str(current), "previous": str(previous)},
    }


def calculate_balance_sheet_millions(question: str, sources: Sequence[dict[str, Any]]):
    """Convert a balance sheet amount only with its OWN page's unit header."""
    if not (re.search(r"\btotal current liabilities\b", question, re.I)
            and re.search(r"\b(?:USD )?millions\b", question, re.I)
            and re.search(r"\bbalance sheet\b", question, re.I)):
        return None
    years = set(_YEAR.findall(question))
    if len(years) != 1:
        return None
    year = years.pop()
    pages: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for source in sources:
        pages.setdefault((source.get("source"), source.get("loader_page_index")), []).append(source)
    candidates: list[tuple[Decimal, str, str]] = []
    for siblings in pages.values():
        page_text = _normal(" ".join(str(s.get("content") or "") for s in siblings))
        if not (re.search(r"CONSOLIDATED BALANCE SHEETS", page_text, re.I)
                and re.search(r"\bin thousands\b", page_text, re.I)
                and re.search(rf"\b{year}\s+(?:19|20)\d{{2}}\b", page_text)):
            continue
        for source in siblings:
            match = re.search(rf"\bTotal current liabilities\s+\$?\s*({_VALUE})", _normal(source.get("content") or ""), re.I)
            if match:
                candidates.append((Decimal(match.group(1).replace(",", "")) / 1000,
                                   source["label"], next(s["label"] for s in siblings if re.search(r"\bin thousands\b", _normal(s.get("content") or ""), re.I))))
    if not candidates or len({a for a, _, _ in candidates}) != 1:
        return None
    value, row_label, unit_label = candidates[0]
    return {"answer": f"FY{year} total current liabilities were ${value:,.3f} million [{row_label}] [{unit_label}].",
            "operands": {"liabilities_in_thousands": str(value * 1000), "liabilities_in_millions": str(value)}}


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
    calculators = (
        ("deterministic_working_capital", calculate_working_capital),
        ("deterministic_credit_capacity", calculate_credit_capacity),
        ("deterministic_revenue_growth", calculate_revenue_growth),
        ("deterministic_balance_sheet_units", calculate_balance_sheet_millions),
    )
    calculation = None
    route = "llm"
    for name, calculator in calculators:
        calculation = calculator(question, sources)
        if calculation is not None:
            route = name
            break
    if calculation is not None:
        candidate = calculation["answer"]
    elif (re.search(r"\btotal current liabilities\b", question, re.I)
          and re.search(r"\bbalance sheet\b", question, re.I)
          and re.search(r"\b(?:USD )?millions\b", question, re.I)):
        # A bare balance-sheet row does not establish its displayed units.
        candidate = "INSUFFICIENT_CONTEXT: balance sheet unit or year header is missing."
        route = "missing_balance_sheet_units"
    else:
        candidate = invoke(build_grounded_prompt(question, context))
    issues = check_answer(candidate, sources, computed=calculation is not None, question=question)
    answer = "INSUFFICIENT_CONTEXT: answer failed citation/provenance checks." if issues else candidate
    return {"answer": answer, "raw_answer": candidate, "route": route, "critic_issues": issues,
            "calculation": calculation["operands"] if calculation else None}
