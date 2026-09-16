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
- **Latency**: retrieval time per question; index construction is excluded.

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

Start with 10 questions to verify the complete pipeline while keeping the first
run short:

```powershell
python .\scripts\evaluate_financebench_retrieval.py --limit 10
```

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
