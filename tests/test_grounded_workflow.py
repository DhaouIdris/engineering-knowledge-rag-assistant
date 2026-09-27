from app.evaluation.grounded_workflow import (
    calculate_working_capital,
    check_answer,
    grounded_answer,
)


def balance_sheet_sources():
    return [
        {"label": "S1", "source": "lockheed.pdf", "loader_page_index": 67,
         "content": "Consolidated Balance Sheets\nDecember 31,\n2021 2020\n"
                    "Total current assets 19,815 19,378"},
        {"label": "S2", "source": "lockheed.pdf", "loader_page_index": 67,
         "content": "Total current liabilities 13,997 13,933"},
    ]


QUESTION = (
    "What is FY2021 net working capital? Define net working capital as "
    "total current assets less total current liabilities. Answer in USD millions."
)


def test_calculator_uses_both_cited_rows_and_first_year():
    result = calculate_working_capital(QUESTION, balance_sheet_sources())
    assert "$5,818 million" in result["answer"]
    assert "[S1]" in result["answer"] and "[S2]" in result["answer"]
    assert result["operands"] == {"assets": "19815", "liabilities": "13997"}


def test_calculator_does_not_guess_missing_or_conflicting_figures():
    assert calculate_working_capital(QUESTION, balance_sheet_sources()[:1]) is None
    conflicting = balance_sheet_sources() + [
        {"label": "S3", "source": "lockheed.pdf", "loader_page_index": 67,
         "content": "Total current liabilities 14,999 13,933"}
    ]
    assert calculate_working_capital(QUESTION, conflicting) is None
    assert calculate_working_capital(QUESTION.replace("FY2021", "FY2020"), balance_sheet_sources()) is None
    assert calculate_working_capital("What is 2021 revenue?", balance_sheet_sources()) is None


def test_calculator_does_not_cross_documents_or_pages():
    sources = balance_sheet_sources()
    sources[1] = {**sources[1], "source": "different.pdf"}
    assert calculate_working_capital(QUESTION, sources) is None


def test_workflow_routes_calculator_without_invoking_llm():
    def fail_if_called(_prompt):
        raise AssertionError("Calculator must not invoke LLM")

    outcome = grounded_answer(QUESTION, "", balance_sheet_sources(), fail_if_called)
    assert outcome["route"] == "deterministic_working_capital"
    assert outcome["critic_issues"] == []
    assert "$5,818 million" in outcome["answer"]


def test_critic_blocks_uncited_or_unsupported_numbers():
    sources = [{"label": "S1", "content": "Cash proceeds were $13.2 billion."}]
    assert "missing_or_invalid_citations" in check_answer("Cash proceeds were $13.2 billion.", sources)
    assert "unsupported_number" in check_answer("Cash proceeds were $99 billion [S1].", sources)
    assert check_answer("Cash proceeds were $13.2 billion [S1].", sources) == []
    outcome = grounded_answer("How much cash?", "context", sources, lambda _: "Cash was $99 billion [S1].")
    assert outcome["route"] == "llm"
    assert outcome["answer"].startswith("INSUFFICIENT_CONTEXT")
    assert "unsupported_number" in outcome["critic_issues"]


def test_critic_does_not_assert_semantic_faithfulness():
    sources = [{"label": "S1", "content": "Revenue increased in FY2022."}]
    # Citation and numeric checks cannot detect the semantic inversion.
    assert check_answer("Revenue decreased in FY2022 [S1].", sources) == []
