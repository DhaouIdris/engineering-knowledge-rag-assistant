from scripts.diagnose_financebench_generation import diagnostics


def test_diagnostics_separates_evidence_page_from_answer_quality():
    source = {"source": "report.pdf", "loader_page_index": 3, "content": "Revenue 5"}
    row = {
        "expected_locations": [{"source": "report.pdf", "loader_page_index": 3}],
        "sources": [source],
        "generation_metrics": {"refusal": 1, "token_f1": 0},
    }
    report = diagnostics({"questions": [row], "configuration": {}})
    assert report["page_present"]["questions"] == 1
    assert report["page_present"]["refusals"] == 1
    assert report["page_absent"]["questions"] == 0
