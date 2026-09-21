# Engineering Knowledge RAG Assistant

A local-first RAG assistant for technical engineering documents. The current
application loads PDFs, splits them into chunks, embeds them with a Sentence
Transformers model, stores them in FAISS, and uses an Ollama-hosted LLM through
LangChain tools.

## Current baseline

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, normalized
- Chunking: 512 characters with 64-character overlap
- Retrieval: MMR, `k=4`, fixed `fetch_k=20`
- Generation: `llama3.2` through Ollama
- UI: Streamlit

## Retrieval evaluation

The repository includes a page-grounded evaluation dataset built from
*Automobile Engineering* by Sudhir Kumar Saxena. The PDF itself is intentionally
not committed. Place a legally obtained copy with this exact filename in
`data/documents/`:

```text
dokumen.pub_a-textbook-of-automobile-engineering-2nbsped-9781944131296-1944131299.pdf
```

Run the baseline:

```bash
python scripts/evaluate_retrieval.py
```

Compare `k=3`, `k=5`, and `k=10` while reusing the same index:

```bash
python scripts/evaluate_retrieval.py --k 3 5 10
```

Compare similarity search with MMR:

```bash
python scripts/evaluate_retrieval.py \
  --k 3 5 10 \
  --search-type similarity mmr
```

Try controlled chunking configurations:

```bash
python scripts/evaluate_retrieval.py \
  --chunk-size 256 512 1024 \
  --chunk-overlap 64 \
  --k 5 \
  --search-type mmr
```

Detailed per-question results are written to
`evaluations/results/retrieval_results.json`.

### Metric definitions

- **Hit@K**: at least one retrieved chunk comes from a relevant source page.
- **Precision@K**: fraction of retrieved chunks whose source page is relevant.
- **Recall@K**: fraction of unique ground-truth pages covered by the retrieved
  chunks. Duplicate overlapping chunks from one page do not inflate recall.
- **MRR**: reciprocal rank of the first chunk from a relevant page, averaged
  across questions.
- **Relaxed Hit/Recall/MRR**: diagnostic variants accepting the same PDF page
  or an adjacent page (`±1` by default). Strict page metrics remain primary.
- **Evidence Hit/Coverage/MRR**: match normalized five-token sequences from
  FinanceBench evidence text against retrieved chunks. This can reveal page
  metadata offsets or extraction effects independently of exact page labels.
- **Latency**: retrieval time per question; index construction is excluded.
- **Document Hit@K / document MRR**: whether the correct PDF is retrieved and
  how highly its first chunk is ranked, independently from exact-page matching.

Detailed FinanceBench output also stores the expected evidence text, a preview
of every retrieved chunk, and its distance from the nearest annotated page.

To isolate passage retrieval from document routing, use the oracle document
scope. FinanceBench supplies the evidence document, so these results must be
reported separately from corpus retrieval:

```powershell
python .\scripts\evaluate_financebench_retrieval.py `
  --limit 10 --smoke-test `
  --retrieval-scope corpus document
```

Run controlled chunking experiments separately so only one parameter pair
changes at a time:

```powershell
python .\scripts\evaluate_financebench_retrieval.py `
  --limit 10 --smoke-test --retrieval-scope document `
  --chunk-size 1024 --chunk-overlap 128 `
  --output evaluations/results/financebench_document_1024.json
```

For BGE v1.5, you can test its recommended query instruction while reusing
the existing BGE index. The prefix is applied only to questions; no PDF is
reloaded or embedded again when the cache matches:

```powershell
python .\scripts\evaluate_financebench_retrieval.py `
  --limit 10 --smoke-test --retrieval-scope document `
  --search-type similarity --embedding-model BAAI/bge-small-en-v1.5 `
  --query-prefix "Represent this sentence for searching relevant passages: " `
  --output evaluations/results/financebench_bge_instruction.json
```

Compare semantic search with a BM25 lexical baseline on the **same chunks**
and the same 10 questions. BM25 matches financial terms and years directly.
The script reads chunks from the existing MiniLM FAISS cache, so this run
does not rebuild the index or download another model:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_financebench_retrieval.py `
  --limit 10 --smoke-test --retrieval-scope document `
  --search-type similarity bm25 --k 3 5 10 `
  --output evaluations/results/financebench_similarity_vs_bm25.json
```

BM25 is an independent lexical baseline here; the script does not combine it
with FAISS scores. In document scope both methods use the FinanceBench evidence
PDF as an oracle filter, so neither score measures document routing.

To check whether lexical and semantic rankings complement each other, fuse
their top 20 candidates using reciprocal rank fusion (RRF). This uses the
existing FAISS cache and BM25 on the same chunks, without a new download:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_financebench_retrieval.py `
  --limit 10 --smoke-test --retrieval-scope document `
  --search-type similarity bm25 rrf --k 3 5 10 `
  --rrf-candidates 20 `
  --output evaluations/results/financebench_rrf_512.json
```

RRF adds `1 / (60 + rank)` per retriever for each chunk. The reported query
latency includes both searches but excludes BM25 index construction. These
ten questions and the oracle document filter provide a quick experiment,
not a full-corpus benchmark; document hit is 1 by construction.

Before scaling the experiment to all 150 questions, screen a stronger compact
embedding model on the same ten questions. BGE applies its retrieval instruction
only to dense queries; BM25 continues to receive the original question:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_financebench_retrieval.py `
  --limit 10 --smoke-test --retrieval-scope document `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --query-prefix "Represent this sentence for searching relevant passages: " `
  --search-type similarity bm25 rrf --k 3 5 10 `
  --rrf-candidates 20 `
  --output evaluations/results/financebench_bge_screen.json
```

Keep BGE for the larger run only if it improves Hit@10 by at least 0.10 or MRR
by at least 0.05 over MiniLM on this fixed smoke set. This is a screening rule,
not a statistical conclusion; final model selection uses all 150 questions.

Ground truth uses source filename plus zero-based PDF page metadata. It does not
use chunk IDs because chunk IDs change when chunk size or overlap changes.

## FinanceBench benchmark data

FinanceBench is the primary external benchmark planned for this project. Its
PDFs and annotations are local evaluation inputs and are intentionally excluded
from Git through `data/benchmarks/`.

Download the official FinanceBench repository as a ZIP, then run the setup
script from the project root in PowerShell:

```powershell
.\scripts\setup_financebench.ps1 `
  -ZipPath "$env:USERPROFILE\Downloads\financebench-main.zip"
```

If PowerShell blocks local scripts, run it once without changing the permanent
execution policy:

```powershell
powershell -ExecutionPolicy Bypass `
  -File .\scripts\setup_financebench.ps1 `
  -ZipPath "$env:USERPROFILE\Downloads\financebench-main.zip"
```

The script extracts the archive to a temporary directory and copies only the
benchmark inputs into this structure:

```text
data/benchmarks/financebench/
├── financebench_open_source.jsonl
├── financebench_document_information.jsonl
└── pdfs/
    └── *.pdf
```

It verifies the number of question records and PDFs after copying. The expected
open-source question count is 150.

### Run the FinanceBench retrieval benchmark

Start with 10 questions and only their referenced PDFs to verify the complete
pipeline quickly:

```powershell
python .\scripts\evaluate_financebench_retrieval.py --limit 10 --smoke-test
```

Smoke-test metrics are intentionally easier and must not be reported as full
FinanceBench results. The smoke index uses a separate cache directory.

Then run all 150 questions and compare similarity search with MMR at three
values of `k`:

```powershell
python .\scripts\evaluate_financebench_retrieval.py
```

The first run loads the 368 PDFs, creates embeddings, and saves a local FAISS
index under `storage/financebench/`. Later runs with the same embedding model,
chunk size, overlap, and unchanged PDF corpus reuse that index. Both the cache
and detailed results are ignored by Git.

Use a fixed MMR candidate pool so changing `k` does not silently change two
variables at once:

```powershell
python .\scripts\evaluate_financebench_retrieval.py `
  --k 3 5 10 `
  --fetch-k 20 `
  --search-type similarity mmr
```

To deliberately rebuild the index after changing the corpus, add
`--rebuild-index`. Results are written to
`evaluations/results/financebench_retrieval.json`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

Create a local `.env` from your own configuration. Never commit `.env` or API
secrets; `.env` is ignored by Git.

## Tests

```bash
pytest -q
```
