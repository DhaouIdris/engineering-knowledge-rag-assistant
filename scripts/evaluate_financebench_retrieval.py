#!/usr/bin/env python3
"""Evaluate retrieval on FinanceBench with reusable FAISS indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.vectorstores import FAISS

from app.core.document_loader import load_documents
from app.core.embeddings import get_embeddings
from app.core.config import settings
from app.evaluation.financebench import load_financebench_dataset
from app.evaluation.retrieval import (
    evaluate_ranked_documents_by_locations,
    mean_metric,
)
from scripts.evaluate_retrieval import build_chunks, make_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FinanceBench retrieval.")
    parser.add_argument("--benchmark-dir", default="data/benchmarks/financebench")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--pdf-dir", default=None)
    parser.add_argument("--chunk-size", nargs="+", type=int, default=[512])
    parser.add_argument("--chunk-overlap", nargs="+", type=int, default=[64])
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument(
        "--search-type", nargs="+", choices=("similarity", "mmr"), default=["similarity", "mmr"]
    )
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--cache-dir", default="storage/financebench")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluations/results/financebench_retrieval.json")
    return parser.parse_args()


def corpus_fingerprint(pdf_dir: Path) -> str:
    """Fingerprint filenames, sizes and modification times without reading huge PDFs."""
    digest = hashlib.sha256()
    for pdf in sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name.lower()):
        stat = pdf.stat()
        digest.update(f"{pdf.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def cache_path(cache_dir: Path, model: str, chunk_size: int, chunk_overlap: int) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_")
    return cache_dir / f"{safe_model}__chunk-{chunk_size}__overlap-{chunk_overlap}"


def load_or_build_store(
    *,
    pages: list[Any] | None,
    embeddings: Any,
    model: str,
    pdf_dir: Path,
    cache_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
) -> tuple[FAISS, bool, int, list[Any] | None]:
    target = cache_path(cache_dir, model, chunk_size, chunk_overlap)
    metadata_path = target / "metadata.json"
    expected = {
        "embedding_model": model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "corpus_fingerprint": corpus_fingerprint(pdf_dir),
        "pdf_count": len(list(pdf_dir.glob("*.pdf"))),
    }

    if not rebuild and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            store = FAISS.load_local(
                str(target), embeddings, allow_dangerous_deserialization=True
            )
            return store, True, int(metadata["chunk_count"]), pages

    if pages is None:
        pages = load_documents(str(pdf_dir))
        if not pages:
            raise ValueError(f"No PDF pages loaded from {pdf_dir}.")
    chunks = build_chunks(pages, chunk_size, chunk_overlap)
    store = FAISS.from_documents(chunks, embeddings)
    target.mkdir(parents=True, exist_ok=True)
    store.save_local(str(target))
    metadata_path.write_text(
        json.dumps({**expected, "chunk_count": len(chunks)}, indent=2), encoding="utf-8"
    )
    return store, False, len(chunks), pages


def evaluate_configuration(
    *, store: FAISS, dataset: dict[str, Any], search_type: str, k: int, fetch_k: int
) -> dict[str, Any]:
    retriever = make_retriever(store, search_type, k, fetch_k)
    rows: list[dict[str, Any]] = []
    for example in dataset["examples"]:
        started = perf_counter()
        documents = retriever.invoke(example["question"])
        latency_ms = (perf_counter() - started) * 1000
        metrics = evaluate_ranked_documents_by_locations(
            documents, relevant_locations=example["relevant_locations"], k=k
        )
        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "question_type": example["question_type"],
                "latency_ms": latency_ms,
                **metrics,
            }
        )

    return {
        "configuration": {
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
    config, summary = result["configuration"], result["summary"]
    print(
        "search={search_type:<10} k={k:<2} | Hit={hit_at_k:.3f} "
        "Precision={precision_at_k:.3f} Recall={recall_at_k:.3f} "
        "MRR={mrr:.3f} Redundancy={redundancy_rate:.3f} "
        "Latency={mean_latency_ms:.1f}ms".format(**config, **summary)
    )


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    questions = Path(args.questions) if args.questions else benchmark_dir / "financebench_open_source.jsonl"
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else benchmark_dir / "pdfs"
    if any(k <= 0 for k in args.k):
        raise SystemExit("Every k value must be strictly positive.")
    if "mmr" in args.search_type and args.fetch_k < max(args.k):
        raise SystemExit("--fetch-k must be greater than or equal to every k value for MMR.")

    dataset = load_financebench_dataset(questions, pdf_dir, limit=args.limit)
    model = args.embedding_model or settings.embedding_model
    embeddings = get_embeddings(args.embedding_model)
    all_results: list[dict[str, Any]] = []
    pages: list[Any] | None = None

    for chunk_size, chunk_overlap in product(args.chunk_size, args.chunk_overlap):
        started = perf_counter()
        store, cache_hit, chunk_count, pages = load_or_build_store(
            pages=pages,
            embeddings=embeddings,
            model=model,
            pdf_dir=pdf_dir,
            cache_dir=Path(args.cache_dir),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild=args.rebuild_index,
        )
        print(
            f"Index {'loaded from cache' if cache_hit else 'built'}: {chunk_count} chunks "
            f"({perf_counter() - started:.1f}s)"
        )
        for search_type, k in product(args.search_type, args.k):
            result = evaluate_configuration(
                store=store, dataset=dataset, search_type=search_type, k=k, fetch_k=args.fetch_k
            )
            result["configuration"].update(
                {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            )
            all_results.append(result)
            print_summary(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": dataset["dataset_name"],
                "embedding_model": model,
                "questions": len(dataset["examples"]),
                "results": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Detailed results written to {output}")


if __name__ == "__main__":
    main()
