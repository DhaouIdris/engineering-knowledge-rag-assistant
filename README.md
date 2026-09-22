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

### Resumable document-scoped evaluation

For the complete 150-question document-scoped evaluation, use independent PDF
indexes instead of building one index for all 368 files. Each PDF is saved as
soon as it is embedded, so an interrupted run resumes from the completed files:

```powershell
python .\scripts\evaluate_financebench_retrieval.py `
  --retrieval-scope document --per-document-cache `
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 `
  --search-type similarity bm25 rrf --k 10 --rrf-candidates 20 `
  --output evaluations/results/financebench_150_diagnostics.json
```

The caches are written under `storage/financebench/documents/`. Do not add
`--rebuild-index` when resuming: valid document indexes will be loaded and only
missing or changed PDFs will be rebuilt.

### Retrieve broadly, then rerank precisely

The dense bi-encoder embeds questions and chunks independently, which makes
FAISS retrieval fast. A cross-encoder is slower but scores each `(question,
chunk)` pair jointly. It is therefore applied only to the top dense candidates:

```text
dense similarity -> top 20 candidates -> cross-encoder -> top 5 or top 10
```

Compare the direct dense baseline with reranking without rebuilding any FAISS
index:

```powershell
python .\scripts\evaluate_financebench_retrieval.py `
  --retrieval-scope document --per-document-cache `
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 `
  --search-type similarity rerank --k 5 10 `
  --rerank-candidates 20 `
  --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2 `
  --rerank-batch-size 16 --reranker-device cpu `
  --page-tolerance 1 --evidence-ngram-size 5 `
  --output evaluations/results/financebench_reranker.json
```

The first reranker run downloads its model through Sentence Transformers. The
result file reports candidate-retrieval latency, reranking latency, and total
latency separately. On a CUDA-enabled machine, `--reranker-device cuda` can be
used instead.

### Evaluate grounded answer generation

Retrieval metrics do not show whether the LLM uses context correctly. The
generation evaluator selects an equal number of `domain-relevant`,
`metrics-generated`, and `novel-generated` questions, retrieves ten chunks, and
asks Ollama for an answer with source labels such as `[S1]`. Because financial
tables are often split across chunks, the generator also adds companion chunks
from the pages represented by the three highest-ranked results. Control this
with `--expand-top-pages` and `--max-context-chunks`; retrieval metrics are still
computed only from the original ranked results.

First verify the complete pipeline on three questions (one per type):

```powershell
python .\scripts\evaluate_financebench_generation.py `
  --samples-per-type 1 --k 10 `
  --ollama-model llama3.2 `
  --checkpoint evaluations/results/generation_smoke_checkpoint.jsonl `
  --output evaluations/results/generation_smoke.json
```

Then evaluate the fixed 30-question stratified sample:

```powershell
python .\scripts\evaluate_financebench_generation.py `
  --samples-per-type 10 --seed 42 --k 10 `
  --ollama-model llama3.2 `
  --checkpoint evaluations/results/financebench_generation_checkpoint.jsonl `
  --output evaluations/results/financebench_generation.json
```

Each completed answer is flushed immediately to the JSONL checkpoint. Re-run
the same command after an interruption to resume. Use `--no-resume` only when
answers should intentionally be regenerated.

The report separates retrieval and generation latency and includes normalized
exact match, token F1, numeric recall, refusal behavior, citation validity,
sentence citation coverage, and strict/relaxed citation agreement with the
FinanceBench evidence pages. These are deterministic diagnostics rather than a
claim of complete semantic faithfulness; qualitative review remains necessary.

For a separate corpus-wide experiment, run all 150 questions and compare
similarity search with MMR at three values of `k`:

```powershell
python .\scripts\evaluate_financebench_retrieval.py
```

This corpus-wide command is intentionally much more expensive: its first run
loads all 368 PDFs and saves one shared FAISS index under
`storage/financebench/full/`. Later runs with the same embedding model, chunk
size, overlap, and unchanged PDF corpus reuse that index. Both caches and
detailed results are ignored by Git.

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
