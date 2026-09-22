#!/usr/bin/env python3
"""Evaluate grounded FinanceBench answer generation with resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import OllamaLLM

from app.core.config import settings
from app.core.embeddings import get_embeddings
from app.evaluation.financebench import load_financebench_dataset
from app.evaluation.generation import (
    build_grounded_prompt,
    evaluate_answer,
    expand_with_same_page_chunks,
    format_context,
    mean_available,
    stratified_sample,
)
from app.evaluation.retrieval import evaluate_ranked_documents_by_locations
from scripts.evaluate_financebench_retrieval import load_or_build_document_stores


PROMPT_VERSION = "financebench_grounded_v3"
PAGE_TOLERANCE = 1
EVIDENCE_NGRAM_SIZE = 5
GENERATION_METRICS = (
    "exact_match",
    "token_f1",
    "numeric_recall",
    "refusal",
    "evidence_action_alignment",
    "citation_presence",
    "citation_validity",
    "citation_sentence_coverage",
    "citation_ground_truth_precision",
    "citation_ground_truth_recall",
    "citation_ground_truth_hit",
    "relaxed_citation_ground_truth_hit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FinanceBench generation.")
    parser.add_argument("--benchmark-dir", default="data/benchmarks/financebench")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--pdf-dir", default=None)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--expand-top-pages", type=int, default=3)
    parser.add_argument("--max-context-chunks", type=int, default=20)
    parser.add_argument("--cache-dir", default="storage/financebench")
    parser.add_argument("--samples-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-model", default=settings.ollama_model)
    parser.add_argument("--ollama-base-url", default=settings.ollama_base_url)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--checkpoint",
        default="evaluations/results/financebench_generation_checkpoint.jsonl",
    )
    parser.add_argument(
        "--output", default="evaluations/results/financebench_generation.json"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard an existing checkpoint and generate every selected answer again.",
    )
    return parser.parse_args()


def run_signature(arguments: argparse.Namespace, selected_ids: list[str]) -> str:
    configuration = {
        "prompt_version": PROMPT_VERSION,
        "embedding_model": arguments.embedding_model,
        "chunk_size": arguments.chunk_size,
        "chunk_overlap": arguments.chunk_overlap,
        "k": arguments.k,
        "expand_top_pages": arguments.expand_top_pages,
        "max_context_chunks": arguments.max_context_chunks,
        "samples_per_type": arguments.samples_per_type,
        "seed": arguments.seed,
        "ollama_model": arguments.ollama_model,
        "temperature": arguments.temperature,
        "num_predict": arguments.num_predict,
        "page_tolerance": PAGE_TOLERANCE,
        "evidence_ngram_size": EVIDENCE_NGRAM_SIZE,
        "selected_ids": selected_ids,
    }
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_checkpoint(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid checkpoint JSON on line {line_number}: {error}"
                ) from error
            if row.get("run_signature") != signature:
                raise ValueError(
                    "Checkpoint configuration differs from this run. Use another "
                    "--checkpoint path or pass --no-resume."
                )
            completed[row["id"]] = row
    return completed


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def invoke_with_retries(llm: OllamaLLM, prompt: str, max_retries: int) -> str:
    for attempt in range(max_retries + 1):
        try:
            return str(llm.invoke(prompt)).strip()
        except Exception:
            if attempt >= max_retries:
                raise
            sleep(2**attempt)
    raise RuntimeError("Unreachable retry state.")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [row["generation_metrics"] for row in rows]
    question_types = sorted({row["question_type"] for row in rows})
    by_type: dict[str, Any] = {}
    for question_type in question_types:
        typed = [
            row["generation_metrics"]
            for row in rows
            if row["question_type"] == question_type
        ]
        by_type[question_type] = {
            "questions": len(typed),
            **{metric: mean_available(typed, metric) for metric in GENERATION_METRICS},
        }

    return {
        "questions": len(rows),
        **{metric: mean_available(metric_rows, metric) for metric in GENERATION_METRICS},
        "context_evidence_hit": mean_available(rows, "context_evidence_hit"),
        "mean_retrieval_latency_ms": mean_available(rows, "retrieval_latency_ms"),
        "mean_generation_latency_ms": mean_available(rows, "generation_latency_ms"),
        "mean_total_latency_ms": mean_available(rows, "total_latency_ms"),
        "by_question_type": by_type,
    }


def print_summary(summary: dict[str, Any]) -> None:
    def display(key: str) -> str:
        value = summary.get(key)
        return "n/a" if value is None else f"{value:.3f}"

    print(
        "Generation summary | "
        f"TokenF1={display('token_f1')} "
        f"NumericRecall={display('numeric_recall')} "
        f"CitationValidity={display('citation_validity')} "
        f"CitationHit={display('citation_ground_truth_hit')} "
        f"Refusal={display('refusal')} "
        f"Latency={display('mean_total_latency_ms')}ms",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be strictly positive.")
    if args.expand_top_pages < 0 or args.max_context_chunks <= 0:
        raise SystemExit(
            "--expand-top-pages must be nonnegative and --max-context-chunks positive."
        )
    if args.samples_per_type <= 0:
        raise SystemExit("--samples-per-type must be strictly positive.")
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        raise SystemExit("Chunk overlap must be nonnegative and smaller than chunk size.")
    if args.num_predict <= 0 or args.max_retries < 0:
        raise SystemExit("--num-predict must be positive and --max-retries nonnegative.")

    benchmark_dir = Path(args.benchmark_dir)
    questions_path = (
        Path(args.questions)
        if args.questions
        else benchmark_dir / "financebench_open_source.jsonl"
    )
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else benchmark_dir / "pdfs"
    dataset = load_financebench_dataset(questions_path, pdf_dir)
    selected = stratified_sample(
        dataset["examples"], args.samples_per_type, args.seed
    )
    selected_dataset = {"dataset_name": dataset["dataset_name"], "examples": selected}
    selected_ids = [example["id"] for example in selected]
    signature = run_signature(args, selected_ids)

    checkpoint_path = Path(args.checkpoint)
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    completed = load_checkpoint(checkpoint_path, signature)
    if completed:
        print(f"Resuming from {len(completed)} completed answers.", flush=True)

    embeddings = get_embeddings(args.embedding_model)
    stores, cache_hits, chunk_count = load_or_build_document_stores(
        dataset=selected_dataset,
        embeddings=embeddings,
        model=args.embedding_model,
        pdf_dir=pdf_dir,
        cache_dir=Path(args.cache_dir) / "documents",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=False,
    )
    print(
        f"Generation indexes ready: {len(stores)} PDFs, {cache_hits} cache hits, "
        f"{chunk_count} chunks.",
        flush=True,
    )
    llm = OllamaLLM(
        base_url=args.ollama_base_url,
        model=args.ollama_model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    for position, example in enumerate(selected, start=1):
        if example["id"] in completed:
            print(
                f"[{position}/{len(selected)}] {example['id']}: checkpoint",
                flush=True,
            )
            continue
        expected_filenames = {
            filename for filename, _ in example["relevant_locations"]
        }
        if len(expected_filenames) != 1:
            raise ValueError("Generation evaluation requires one evidence PDF.")
        expected_filename = next(iter(expected_filenames))

        started = perf_counter()
        documents = stores[expected_filename].similarity_search(
            example["question"], k=args.k
        )
        retrieval_latency_ms = (perf_counter() - started) * 1000
        retrieval_metrics = evaluate_ranked_documents_by_locations(
            documents,
            relevant_locations=example["relevant_locations"],
            k=args.k,
            page_tolerance=PAGE_TOLERANCE,
            evidence_texts=[
                (item["source"], item["evidence_text"])
                for item in example["evidence"]
                if item.get("evidence_text")
            ],
            evidence_ngram_size=EVIDENCE_NGRAM_SIZE,
        )
        context_documents = expand_with_same_page_chunks(
            documents,
            stores[expected_filename],
            top_pages=args.expand_top_pages,
            max_documents=args.max_context_chunks,
        )
        context, sources = format_context(context_documents)
        prompt = build_grounded_prompt(example["question"], context)

        generation_started = perf_counter()
        answer = invoke_with_retries(llm, prompt, args.max_retries)
        generation_latency_ms = (perf_counter() - generation_started) * 1000
        generation_metrics = evaluate_answer(
            answer,
            str(example.get("answer") or ""),
            sources,
            example["relevant_locations"],
            context_evidence_hit=bool(retrieval_metrics["evidence_hit_at_k"]),
        )
        row = {
            "run_signature": signature,
            "id": example["id"],
            "question_type": example["question_type"],
            "question_reasoning": example["question_reasoning"],
            "question": example["question"],
            "reference_answer": example["answer"],
            "reference_justification": example["justification"],
            "answer": answer,
            "sources": sources,
            "expected_locations": [
                {"source": filename, "loader_page_index": page}
                for filename, page in sorted(example["relevant_locations"])
            ],
            "context_hit": retrieval_metrics["hit_at_k"],
            "context_recall": retrieval_metrics["recall_at_k"],
            "context_evidence_hit": retrieval_metrics["evidence_hit_at_k"],
            "generation_metrics": generation_metrics,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": retrieval_latency_ms + generation_latency_ms,
        }
        append_checkpoint(checkpoint_path, row)
        completed[example["id"]] = row
        print(
            f"[{position}/{len(selected)}] {example['id']}: "
            f"F1={generation_metrics['token_f1']:.3f} "
            f"CitationHit={generation_metrics['citation_ground_truth_hit']:.0f} "
            f"Evidence={retrieval_metrics['evidence_hit_at_k']:.0f} "
            f"generation={generation_latency_ms / 1000:.1f}s",
            flush=True,
        )

    rows = [completed[example_id] for example_id in selected_ids]
    configuration = {
        "prompt_version": PROMPT_VERSION,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "search_type": "similarity",
        "k": args.k,
        "expand_top_pages": args.expand_top_pages,
        "max_context_chunks": args.max_context_chunks,
        "samples_per_type": args.samples_per_type,
        "seed": args.seed,
        "ollama_model": args.ollama_model,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "page_tolerance": PAGE_TOLERANCE,
        "evidence_ngram_size": EVIDENCE_NGRAM_SIZE,
        "run_signature": signature,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    output_path.write_text(
        json.dumps(
            {
                "dataset": dataset["dataset_name"],
                "configuration": configuration,
                "summary": summary,
                "questions": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary)
    print(f"Generation results written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
