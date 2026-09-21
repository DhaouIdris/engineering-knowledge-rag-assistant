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
from app.evaluation.fusion import reciprocal_rank_fusion
from app.evaluation.lexical import BM25Index
from app.evaluation.reranking import CrossEncoderReranker
from app.evaluation.retrieval import (
    evaluate_ranked_documents_by_locations,
    mean_metric,
    source_basename,
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
        "--search-type",
        nargs="+",
        choices=("similarity", "mmr", "bm25", "rrf", "rerank"),
        default=["similarity", "mmr"],
    )
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument(
        "--rrf-candidates", type=int, default=20,
        help="Candidates from each of FAISS and BM25 before reciprocal rank fusion.",
    )
    parser.add_argument("--page-tolerance", type=int, default=1)
    parser.add_argument("--evidence-ngram-size", type=int, default=5)
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument(
        "--retrieval-scope",
        nargs="+",
        choices=("corpus", "document"),
        default=["corpus"],
        help="Search the corpus or oracle-filter to the FinanceBench evidence document.",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--query-prefix",
        default="",
        help="Prefix applied to retrieval queries only, without rebuilding document embeddings.",
    )
    parser.add_argument("--cache-dir", default="storage/financebench")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--per-document-cache",
        action="store_true",
        help=(
            "For document-scope evaluation, build and persist one FAISS index per "
            "evidence PDF. Completed PDFs are reusable after an interrupted run."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Index only PDFs referenced by the limited questions. Not a comparable benchmark run.",
    )
    parser.add_argument("--output", default="evaluations/results/financebench_retrieval.json")
    return parser.parse_args()


def selected_pdfs(pdf_dir: Path, filenames: set[str] | None) -> list[Path]:
    return sorted(
        (pdf for pdf in pdf_dir.glob("*.pdf") if filenames is None or pdf.name in filenames),
        key=lambda path: path.name.lower(),
    )


def corpus_fingerprint(pdf_dir: Path, filenames: set[str] | None = None) -> str:
    """Fingerprint filenames, sizes and modification times without reading huge PDFs."""
    digest = hashlib.sha256()
    for pdf in selected_pdfs(pdf_dir, filenames):
        stat = pdf.stat()
        digest.update(f"{pdf.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def cache_path(cache_dir: Path, model: str, chunk_size: int, chunk_overlap: int) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_")
    return cache_dir / f"{safe_model}__chunk-{chunk_size}__overlap-{chunk_overlap}"


def document_cache_path(
    cache_dir: Path,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    filename: str,
) -> Path:
    """Return a stable, collision-resistant cache path for one PDF."""
    root = cache_path(cache_dir, model, chunk_size, chunk_overlap)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(filename).stem).strip("_")
    suffix = hashlib.sha256(filename.encode()).hexdigest()[:10]
    return root / f"{safe_name}__{suffix}"


def load_or_build_document_stores(
    *,
    dataset: dict[str, Any],
    embeddings: Any,
    model: str,
    pdf_dir: Path,
    cache_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
) -> tuple[dict[str, FAISS], int, int]:
    """Load/build independent indexes, saving every PDF as soon as it finishes."""
    filenames = sorted(
        {
            filename
            for example in dataset["examples"]
            for filename, _ in example["relevant_locations"]
        },
        key=str.lower,
    )
    stores: dict[str, FAISS] = {}
    cache_hits = 0
    total_chunks = 0

    for position, filename in enumerate(filenames, start=1):
        started = perf_counter()
        target = document_cache_path(
            cache_dir, model, chunk_size, chunk_overlap, filename
        )
        metadata_path = target / "metadata.json"
        expected = {
            "embedding_model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "corpus_fingerprint": corpus_fingerprint(pdf_dir, {filename}),
            "pdf_count": 1,
            "document": filename,
        }
        cache_hit = False

        if not rebuild and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(metadata.get(key) == value for key, value in expected.items()):
                store = FAISS.load_local(
                    str(target), embeddings, allow_dangerous_deserialization=True
                )
                chunk_count = int(metadata["chunk_count"])
                cache_hit = True

        if not cache_hit:
            pages = load_documents(str(pdf_dir), filenames={filename})
            if not pages:
                raise ValueError(f"No PDF pages loaded for {filename}.")
            chunks = build_chunks(pages, chunk_size, chunk_overlap)
            print(
                f"[{position}/{len(filenames)}] Embedding {filename}: "
                f"{len(pages)} pages, {len(chunks)} chunks...",
                flush=True,
            )
            store = FAISS.from_documents(chunks, embeddings)
            chunk_count = len(chunks)
            target.mkdir(parents=True, exist_ok=True)
            store.save_local(str(target))
            metadata_path.write_text(
                json.dumps({**expected, "chunk_count": chunk_count}, indent=2),
                encoding="utf-8",
            )
        else:
            cache_hits += 1

        stores[filename] = store
        total_chunks += chunk_count
        state = "cache" if cache_hit else "built"
        print(
            f"[{position}/{len(filenames)}] {filename}: {state}, {chunk_count} chunks "
            f"({perf_counter() - started:.1f}s)",
            flush=True,
        )

    return stores, cache_hits, total_chunks


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
    filenames: set[str] | None,
) -> tuple[FAISS, bool, int, list[Any] | None]:
    target = cache_path(cache_dir, model, chunk_size, chunk_overlap)
    metadata_path = target / "metadata.json"
    expected = {
        "embedding_model": model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "corpus_fingerprint": corpus_fingerprint(pdf_dir, filenames),
        "pdf_count": len(selected_pdfs(pdf_dir, filenames)),
    }

    if not rebuild and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            store = FAISS.load_local(
                str(target), embeddings, allow_dangerous_deserialization=True
            )
            return store, True, int(metadata["chunk_count"]), pages

    if pages is None:
        pages = load_documents(str(pdf_dir), filenames=filenames)
        if not pages:
            raise ValueError(f"No PDF pages loaded from {pdf_dir}.")
    print(
        f"Splitting {len(pages)} pages with chunk_size={chunk_size}, "
        f"overlap={chunk_overlap}...",
        flush=True,
    )
    chunks = build_chunks(pages, chunk_size, chunk_overlap)
    print(f"Embedding {len(chunks)} chunks and building FAISS index...", flush=True)
    store = FAISS.from_documents(chunks, embeddings)
    target.mkdir(parents=True, exist_ok=True)
    store.save_local(str(target))
    metadata_path.write_text(
        json.dumps({**expected, "chunk_count": len(chunks)}, indent=2), encoding="utf-8"
    )
    return store, False, len(chunks), pages


def evaluate_configuration(
    *,
    store: FAISS | None,
    document_stores: dict[str, FAISS] | None = None,
    dataset: dict[str, Any],
    search_type: str,
    k: int,
    fetch_k: int,
    retrieval_scope: str,
    query_prefix: str = "",
    rrf_candidates: int = 20,
    page_tolerance: int = 1,
    evidence_ngram_size: int = 5,
    reranker: CrossEncoderReranker | None = None,
    rerank_candidates: int = 20,
) -> dict[str, Any]:
    retriever = None
    source_lookup: dict[str, str] = {}
    lexical_index: BM25Index | None = None
    lexical_by_document: dict[str, BM25Index] = {}
    if search_type in ("bm25", "rrf"):
        if document_stores is not None:
            lexical_by_document = {
                filename: BM25Index(list(document_store.docstore._dict.values()))
                for filename, document_store in document_stores.items()
            }
        elif retrieval_scope == "corpus":
            if store is None:
                raise ValueError("Corpus retrieval requires a shared index.")
            chunks = list(store.docstore._dict.values())
            lexical_index = BM25Index(chunks)
        else:
            if store is None:
                raise ValueError("Document retrieval requires an index.")
            chunks = list(store.docstore._dict.values())
            grouped: dict[str, list[Any]] = {}
            for chunk in chunks:
                filename = source_basename(chunk.metadata.get("source"))
                grouped.setdefault(filename, []).append(chunk)
            lexical_by_document = {
                filename: BM25Index(group) for filename, group in grouped.items()
            }
    if retrieval_scope == "corpus":
        if store is None:
            raise ValueError("Corpus retrieval requires a shared index.")
        if search_type not in ("bm25", "rrf", "rerank"):
            retriever = make_retriever(store, search_type, k, fetch_k)
    elif document_stores is None:
        if store is None:
            raise ValueError("Document retrieval requires an index.")
        for document in store.docstore._dict.values():
            source = str(document.metadata.get("source", ""))
            source_lookup[source_basename(source)] = source

    rows: list[dict[str, Any]] = []
    for example in dataset["examples"]:
        retrieval_query = query_prefix + example["question"]
        started = perf_counter()
        if retrieval_scope == "corpus":
            if search_type == "rrf":
                dense = store.similarity_search(retrieval_query, k=rrf_candidates)
                lexical = lexical_index.search(example["question"], rrf_candidates)
                documents = reciprocal_rank_fusion((dense, lexical), k=k)
            elif search_type == "bm25":
                documents = lexical_index.search(example["question"], k)
            elif search_type == "rerank":
                documents = store.similarity_search(
                    retrieval_query, k=rerank_candidates
                )
            else:
                documents = retriever.invoke(retrieval_query)
        else:
            expected_filenames = {
                filename for filename, _ in example["relevant_locations"]
            }
            if len(expected_filenames) != 1:
                raise ValueError(
                    "Document-scope evaluation currently requires exactly one evidence PDF."
                )
            expected_filename = next(iter(expected_filenames))
            document_store = (
                document_stores.get(expected_filename)
                if document_stores is not None
                else None
            )
            if document_stores is not None and document_store is None:
                raise ValueError(f"Indexed PDF not found: {expected_filename}")
            if search_type in ("bm25", "rrf") and expected_filename not in lexical_by_document:
                raise ValueError(f"Indexed PDF not found: {expected_filename}")
            if search_type == "bm25":
                documents = lexical_by_document[expected_filename].search(example["question"], k)
            elif document_stores is None and expected_filename not in source_lookup:
                raise ValueError(f"Indexed PDF not found: {expected_filename}")
            else:
                active_store = document_store if document_store is not None else store
                if search_type in ("similarity", "rrf", "rerank"):
                    if document_store is not None:
                        dense = active_store.similarity_search(
                            retrieval_query,
                            k=(
                                rrf_candidates
                                if search_type == "rrf"
                                else rerank_candidates
                                if search_type == "rerank"
                                else k
                            ),
                        )
                    else:
                        metadata_filter = {"source": source_lookup[expected_filename]}
                        filter_fetch_k = int(active_store.index.ntotal)
                        dense = active_store.similarity_search(
                            retrieval_query,
                            k=(
                                rrf_candidates
                                if search_type == "rrf"
                                else rerank_candidates
                                if search_type == "rerank"
                                else k
                            ),
                            filter=metadata_filter,
                            fetch_k=filter_fetch_k,
                        )
                    if search_type == "rrf":
                        lexical = lexical_by_document[expected_filename].search(
                            example["question"], rrf_candidates
                        )
                        documents = reciprocal_rank_fusion((dense, lexical), k=k)
                    else:
                        documents = dense
                else:
                    if document_store is not None:
                        documents = active_store.max_marginal_relevance_search(
                            retrieval_query, k=k, fetch_k=fetch_k
                        )
                    else:
                        metadata_filter = {"source": source_lookup[expected_filename]}
                        filter_fetch_k = int(active_store.index.ntotal)
                        documents = active_store.max_marginal_relevance_search(
                            retrieval_query,
                            k=k,
                            filter=metadata_filter,
                            fetch_k=filter_fetch_k,
                        )
        retrieval_latency_ms = (perf_counter() - started) * 1000
        reranking_latency_ms = 0.0
        if search_type == "rerank":
            if reranker is None:
                raise ValueError("Rerank search requires a cross-encoder reranker.")
            reranking_started = perf_counter()
            documents = reranker.rerank(example["question"], documents, k=k)
            reranking_latency_ms = (perf_counter() - reranking_started) * 1000
        latency_ms = retrieval_latency_ms + reranking_latency_ms
        metrics = evaluate_ranked_documents_by_locations(
            documents,
            relevant_locations=example["relevant_locations"],
            k=k,
            page_tolerance=page_tolerance,
            evidence_texts=[
                (item["source"], item["evidence_text"])
                for item in example["evidence"]
                if item.get("evidence_text")
            ],
            evidence_ngram_size=evidence_ngram_size,
        )
        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "justification": example["justification"],
                "question_type": example["question_type"],
                "question_reasoning": example["question_reasoning"],
                "evidence": example["evidence"],
                "latency_ms": latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "reranking_latency_ms": reranking_latency_ms,
                **metrics,
            }
        )

    return {
        "configuration": {
            "search_type": search_type,
            "k": k,
            "fetch_k": fetch_k if search_type == "mmr" else None,
            "rrf_candidates": rrf_candidates if search_type == "rrf" else None,
            "rrf_constant": 60 if search_type == "rrf" else None,
            "reranker_model": reranker.model_name if search_type == "rerank" else None,
            "rerank_candidates": rerank_candidates if search_type == "rerank" else None,
            "page_tolerance": page_tolerance,
            "evidence_ngram_size": evidence_ngram_size,
            "retrieval_scope": retrieval_scope,
            "query_prefix": query_prefix,
        },
        "summary": {
            "questions": len(rows),
            "hit_at_k": mean_metric(rows, "hit_at_k"),
            "precision_at_k": mean_metric(rows, "precision_at_k"),
            "unique_page_precision_at_k": mean_metric(rows, "unique_page_precision_at_k"),
            "recall_at_k": mean_metric(rows, "recall_at_k"),
            "mrr": mean_metric(rows, "reciprocal_rank"),
            "relaxed_hit_at_k": mean_metric(rows, "relaxed_hit_at_k"),
            "relaxed_recall_at_k": mean_metric(rows, "relaxed_recall_at_k"),
            "relaxed_mrr": mean_metric(rows, "relaxed_reciprocal_rank"),
            "evidence_hit_at_k": mean_metric(rows, "evidence_hit_at_k"),
            "evidence_coverage_at_k": mean_metric(rows, "evidence_coverage_at_k"),
            "evidence_mrr": mean_metric(rows, "evidence_reciprocal_rank"),
            "document_hit_at_k": mean_metric(rows, "document_hit_at_k"),
            "document_mrr": mean_metric(rows, "document_reciprocal_rank"),
            "redundancy_rate": mean_metric(rows, "redundancy_rate"),
            "mean_latency_ms": mean_metric(rows, "latency_ms"),
            "mean_retrieval_latency_ms": mean_metric(rows, "retrieval_latency_ms"),
            "mean_reranking_latency_ms": mean_metric(rows, "reranking_latency_ms"),
        },
        "questions": rows,
    }


def print_summary(result: dict[str, Any]) -> None:
    config, summary = result["configuration"], result["summary"]
    line = (
        "scope={retrieval_scope:<8} search={search_type:<10} k={k:<2} | Hit={hit_at_k:.3f} "
        "Precision={precision_at_k:.3f} Recall={recall_at_k:.3f} "
        "MRR={mrr:.3f} RelaxedHit={relaxed_hit_at_k:.3f} "
        "EvidenceHit={evidence_hit_at_k:.3f} DocHit={document_hit_at_k:.3f} "
        "DocMRR={document_mrr:.3f} Redundancy={redundancy_rate:.3f} "
        "Latency={mean_latency_ms:.1f}ms".format(**config, **summary)
    )
    if config["search_type"] == "rerank":
        line += (
            " (retrieval={mean_retrieval_latency_ms:.1f}ms, "
            "rerank={mean_reranking_latency_ms:.1f}ms)"
        ).format(**summary)
    print(line)


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    questions = Path(args.questions) if args.questions else benchmark_dir / "financebench_open_source.jsonl"
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else benchmark_dir / "pdfs"
    if any(k <= 0 for k in args.k):
        raise SystemExit("Every k value must be strictly positive.")
    if args.page_tolerance < 0:
        raise SystemExit("--page-tolerance must be nonnegative.")
    if args.evidence_ngram_size <= 0:
        raise SystemExit("--evidence-ngram-size must be strictly positive.")
    if "mmr" in args.search_type and args.fetch_k < max(args.k):
        raise SystemExit("--fetch-k must be greater than or equal to every k value for MMR.")
    if "rrf" in args.search_type and args.rrf_candidates < max(args.k):
        raise SystemExit("--rrf-candidates must be greater than or equal to every k value for RRF.")
    if "rerank" in args.search_type and args.rerank_candidates < max(args.k):
        raise SystemExit("--rerank-candidates must be greater than or equal to every k value.")
    if args.rerank_batch_size <= 0:
        raise SystemExit("--rerank-batch-size must be strictly positive.")
    if args.smoke_test and args.limit is None:
        raise SystemExit("--smoke-test requires --limit so it cannot be mistaken for a full benchmark.")
    if args.per_document_cache and set(args.retrieval_scope) != {"document"}:
        raise SystemExit("--per-document-cache requires --retrieval-scope document only.")
    if set(args.search_type) == {"bm25"} and args.query_prefix:
        raise SystemExit("--query-prefix has no effect when BM25 is the only search type.")

    dataset = load_financebench_dataset(questions, pdf_dir, limit=args.limit)
    filenames = None
    cache_dir = Path(args.cache_dir) / "full"
    if args.smoke_test:
        filenames = {
            filename
            for example in dataset["examples"]
            for filename, _ in example["relevant_locations"]
        }
        cache_dir = Path(args.cache_dir) / "smoke"
        print(
            f"SMOKE TEST: indexing {len(filenames)} relevant PDFs for "
            f"{len(dataset['examples'])} questions. Results are not comparable "
            "to the full-corpus benchmark.",
            flush=True,
        )
    model = args.embedding_model or settings.embedding_model
    embeddings = get_embeddings(args.embedding_model)
    reranker = None
    if "rerank" in args.search_type:
        print(f"Loading reranker model: {args.reranker_model}", flush=True)
        reranker = CrossEncoderReranker(
            args.reranker_model,
            batch_size=args.rerank_batch_size,
            device=args.reranker_device,
        )
    all_results: list[dict[str, Any]] = []
    pages: list[Any] | None = None

    for chunk_size, chunk_overlap in product(args.chunk_size, args.chunk_overlap):
        started = perf_counter()
        document_stores: dict[str, FAISS] | None = None
        document_cache_hits: int | None = None
        if args.per_document_cache:
            document_stores, document_cache_hits, chunk_count = (
                load_or_build_document_stores(
                    dataset=dataset,
                    embeddings=embeddings,
                    model=model,
                    pdf_dir=pdf_dir,
                    cache_dir=Path(args.cache_dir) / "documents",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    rebuild=args.rebuild_index,
                )
            )
            store = None
            print(
                f"Document indexes ready: {len(document_stores)} PDFs, "
                f"{document_cache_hits} cache hits, {chunk_count} chunks "
                f"({perf_counter() - started:.1f}s)"
            )
        else:
            store, cache_hit, chunk_count, pages = load_or_build_store(
                pages=pages,
                embeddings=embeddings,
                model=model,
                pdf_dir=pdf_dir,
                cache_dir=cache_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                rebuild=args.rebuild_index,
                filenames=filenames,
            )
            print(
                f"Index {'loaded from cache' if cache_hit else 'built'}: "
                f"{chunk_count} chunks ({perf_counter() - started:.1f}s)"
            )
        for retrieval_scope, search_type, k in product(
            args.retrieval_scope, args.search_type, args.k
        ):
            result = evaluate_configuration(
                store=store,
                document_stores=document_stores,
                dataset=dataset,
                search_type=search_type,
                k=k,
                fetch_k=args.fetch_k,
                retrieval_scope=retrieval_scope,
                query_prefix=args.query_prefix,
                rrf_candidates=args.rrf_candidates,
                page_tolerance=args.page_tolerance,
                evidence_ngram_size=args.evidence_ngram_size,
                reranker=reranker,
                rerank_candidates=args.rerank_candidates,
            )
            result["configuration"].update(
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "index_strategy": (
                        "per_document" if document_stores is not None else "shared"
                    ),
                    "document_index_cache_hits": document_cache_hits,
                }
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
