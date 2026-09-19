from dataclasses import dataclass

from app.evaluation.lexical import BM25Index, tokenize


@dataclass
class Chunk:
    page_content: str


def test_exact_fiscal_year_separates_similar_financial_passages():
    wrong_year = Chunk("2019 consolidated cash flow. Capital expenditures were 1,490.")
    correct_year = Chunk("2018 consolidated cash flow. Capital expenditures were 1,577.")
    index = BM25Index([wrong_year, correct_year])

    assert tokenize("FY2018 spending 1,577") == ["2018", "spending", "1577"]
    assert index.search("FY2018 capital expenditures", k=1) == [correct_year]


def test_unmatched_query_returns_no_arbitrary_chunks():
    index = BM25Index([Chunk("balance sheet for 2022")])
    assert index.search("dividends", k=3) == []
