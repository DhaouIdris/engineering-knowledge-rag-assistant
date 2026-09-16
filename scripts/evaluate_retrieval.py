#!/usr/bin/env python3
"""Run reproducible retrieval-only RAG experiments.

Example:
    python scripts/evaluate_retrieval.py \
      --documents data/documents \
      --dataset evaluations/datasets/automobile_engineering_v1.json \
      --k 3 5 10 \
      --search-type similarity mmr
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.embeddings import get_embeddings
from app.core.document_loader import load_documents
from app.evaluation.retrieval import (
    evaluate_ranked_documents,
    load_evaluation_dataset,
    mean_metric,
    relevant_page_indices,
    source_basename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS retrieval with page-grounded Hit@K, Precision@K, Recall@K and MRR."
    )
    parser.add_argument("--documents", default="data/documents", help="Folder containing PDFs.")
    parser.add_argument(
        "--dataset",
        default="evaluations/datasets/automobile_engineering_v1.json",
        help="Evaluation dataset JSON path.",
    )
    parser.add_argument("--chunk-size", nargs="+", type=int, default=[512])
    parser.add_argument("--chunk-overlap", nargs="+", type=int, default=[64])
    parser.add_argument("--k", nargs="+", type=int, default=[4])
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
        help="Fixed MMR candidate-pool size, shared across all k values.",
    )
    parser.add_argument(
        "--search-type",
        nargs="+",
        choices=("similarity", "mmr"),
        default=["mmr"],
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Hugging Face embedding model. Defaults to the application setting.",
    )
    parser.add_argument(
        "--output",
        default="evaluations/results/retrieval_results.json",
        help="Where to write detailed JSON results.",
    )
    return parser.parse_args()


def build_chunks(pages: list[Any], chunk_size: int, chunk_overlap: int) -> list[Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be strictly positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(pages)


def make_retriever(store: FAISS, search_type: str, k: int, fetch_k: int = 20):
    search_kwargs: dict[str, int] = {"k": k}
    if search_type == "mmr":
        if fetch_k < k:
            raise ValueError("fetch_k must be greater than or equal to k for MMR.")
        search_kwargs["fetch_k"] = fetch_k
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def evaluate_configuration(
    *,
    store: FAISS,
    dataset: dict[str, Any],
    search_type: str,
    k: int,
    chunk_size: int,
    chunk_overlap: int,
    fetch_k: int,
) -> dict[str, Any]:
    retriever = make_retriever(store, search_type, k, fetch_k)
    source_filename = dataset["source_document"]["filename"]
    rows: list[dict[str, Any]] = []

    for example in dataset["examples"]:
        started = perf_counter()
        documents = retriever.invoke(example["question"])
        latency_ms = (perf_counter() - started) * 1000

        metrics = evaluate_ranked_documents(
            documents,
            relevant_pages=relevant_page_indices(example),
            expected_filename=source_filename,
            k=k,
        )
        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "latency_ms": latency_ms,
                **metrics,
            }
        )

    return {
        "configuration": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "search_type": search_type,
            "k": k,
            "fetch_k": fetch_k if search_type == "mmr" else None,
        },
        "summary": {
            "questions": len(rows),
            "hit_at_k": mean_metric(rows, "hit_at_k"),
            "precision_at_k": mean_metric(rows, "precision_at_k"),
            "unique_page_precision_at_k": mean_metric(rows, "unique_page_precision_at_k"),
            "recall_at_k": mean_metric(rows, "recall_at_k"),
            "mrr": mean_metric(rows, "reciprocal_rank"),
            "redundancy_rate": mean_metric(rows, "redundancy_rate"),
            "mean_latency_ms": mean_metric(rows, "latency_ms"),
        },
        "questions": rows,
    }


def print_summary(result: dict[str, Any]) -> None:
    config = result["configuration"]
    summary = result["summary"]
    print(
        "chunk={chunk_size:<4} overlap={chunk_overlap:<4} "
        "search={search_type:<10} k={k:<2} | "
        "Hit={hit_at_k:.3f} Precision={precision_at_k:.3f} "
        "Recall={recall_at_k:.3f} MRR={mrr:.3f} "
        "Latency={mean_latency_ms:.1f}ms".format(**config, **summary)
    )


def main() -> None:
    args = parse_args()
    dataset = load_evaluation_dataset(args.dataset)
    loaded_pages = load_documents(args.documents)
    if not loaded_pages:
        raise SystemExit(f"No PDF pages loaded from {args.documents!r}.")

    expected_pdf = dataset["source_document"]["filename"]
    pages = [
        page
        for page in loaded_pages
        if source_basename(page.metadata.get("source")) == expected_pdf
    ]
    if not pages:
        raise SystemExit(
            f"The dataset expects {expected_pdf!r}, but it was not found in {args.documents!r}."
        )

    embeddings = get_embeddings(args.embedding_model)
    results: list[dict[str, Any]] = []

    for chunk_size, chunk_overlap in product(args.chunk_size, args.chunk_overlap):
        chunks = build_chunks(pages, chunk_size, chunk_overlap)
        store = FAISS.from_documents(chunks, embeddings)

        for search_type, k in product(args.search_type, args.k):
            result = evaluate_configuration(
                store=store,
                dataset=dataset,
                search_type=search_type,
                k=k,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                fetch_k=args.fetch_k,
            )
            results.append(result)
            print_summary(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": dataset["dataset_name"],
                "embedding_model": args.embedding_model or "application_default",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Detailed results written to {output}")


if __name__ == "__main__":
    main()
