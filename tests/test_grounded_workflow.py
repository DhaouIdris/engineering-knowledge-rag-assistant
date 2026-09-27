from app.evaluation.grounded_workflow import (
    calculate_balance_sheet_millions,
    calculate_credit_capacity,
    calculate_revenue_growth,
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


def test_credit_capacity_adds_distinct_active_facilities():
    question = "As of May 26, 2023, what is the total amount a company may borrow under its unsecured revolving credit agreements?"
    sources = [
        {"label": "S1", "source": "filing.pdf", "loader_page_index": 1,
         "content": "On May 26, 2023, Acme entered into a new $4,200,000,000 five year unsecured revolving credit agreement."
                    " Acme may increase commitments to $4,950,000,000."},
        {"label": "S2", "source": "filing.pdf", "loader_page_index": 1,
         "content": "On May 26, 2023, Acme entered into a new $4,200,000,000 364 day unsecured revolving credit agreement."},
        {"label": "S3", "source": "filing.pdf", "loader_page_index": 1,
         "content": "Acme terminated its $3,800,000,000 364 day agreement."},
    ]
    assert "$8,400,000,000" in calculate_credit_capacity(question, sources)["answer"]
    assert calculate_credit_capacity(question, sources[:1]) is None
    assert calculate_credit_capacity(question.replace("May 26", "May 27"), sources) is None
    assert calculate_credit_capacity(question, sources + [{**sources[1], "label": "S4", "content": sources[1]["content"].replace("4,200", "4,300")}]) is None


def test_revenue_growth_uses_income_statement_and_requested_precision():
    question = "What is the FY2019 - FY2020 total revenue growth rate? Answer in percents and round to one decimal place."
    sources = [
        {"label": "S1", "content": "Results of operations. Total net revenue $9,497,578 $4,713,500 101%"},
        {"label": "S2", "content": "CONSOLIDATED STATEMENTS OF\n OPERATIONS (in thousands)\n"
                                  "2020\n2019 2018 Revenue: Total net revenue\n9,497,578 4,713,500 3,298,177"},
    ]
    result = calculate_revenue_growth(question, sources)
    assert "101.5%" in result["answer"] and "[S2]" in result["answer"]
    assert calculate_revenue_growth(question, sources[:1]) is None
    assert calculate_revenue_growth(question.replace("FY2019", "FY2018"), sources) is None


def test_balance_sheet_units_require_own_page_header():
    question = "What are FY2017 total current liabilities in USD millions? Base your answer on the balance sheet."
    row = {"label": "S2", "source": "report.pdf", "loader_page_index": 44,
           "content": "Current liabilities: Total current liabilities 5,466,312 4,586,657"}
    header = {"label": "S1", "source": "report.pdf", "loader_page_index": 44,
              "content": "CONSOLIDATED BALANCE SHEETS (in thousands) December 31, 2017 2016"}
    assert calculate_balance_sheet_millions(question, [row]) is None
    result = calculate_balance_sheet_millions(question, [header, row])
    assert "$5,466.312 million" in result["answer"]
    assert "[S1]" in result["answer"] and "[S2]" in result["answer"]
    assert calculate_balance_sheet_millions(question, [{**header, "loader_page_index": 43}, row]) is None
    outcome = grounded_answer(question, "", [row], lambda _: "Current liabilities were $5,466,312 [S2].")
    assert outcome["route"] == "missing_balance_sheet_units"
    assert outcome["answer"].startswith("INSUFFICIENT_CONTEXT")
