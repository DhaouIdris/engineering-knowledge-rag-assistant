#!/usr/bin/env python3
"""Read saved generation results without loading PDFs, embeddings or Ollama."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def diagnostics(data: dict) -> dict:
    rows = data["questions"]
    if not rows:
        raise ValueError("No evaluated questions.")
    groups = {"page_present": [], "page_absent": []}
    for row in rows:
        expected = {(item["source"], item["loader_page_index"])
                    for item in row["expected_locations"]}
        actual = {(item["source"], item["loader_page_index"])
                  for item in row["sources"]}
        groups["page_present" if expected & actual else "page_absent"].append(row)

    def stats(group: list[dict]) -> dict:
        if not group:
            return {"questions": 0}
        return {
            "questions": len(group),
            "refusals": sum(bool(row["generation_metrics"]["refusal"]) for row in group),
            "mean_token_f1": round(statistics.mean(row["generation_metrics"]["token_f1"]
                                                    for row in group), 3),
            "mean_context_chunks": round(statistics.mean(len(row["sources"]) for row in group), 1),
            "mean_source_characters": round(statistics.mean(
                sum(len(source["content"]) for source in row["sources"])
                for row in group)),
            "mean_prompt_characters": (
                round(statistics.mean(row["prompt_characters"] for row in group))
                if all("prompt_characters" in row for row in group) else None
            ),
        }

    return {
        "scope": data.get("configuration", {}).get("pdf_scope",
                 "oracle_document_from_benchmark_annotation"),
        "context_mode": data.get("configuration", {}).get("context_mode", "retrieved"),
        "all": stats(rows),
        **{name: stats(group) for name, group in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = diagnostics(json.loads(args.result.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
