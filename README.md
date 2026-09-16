# Engineering Knowledge RAG Assistant

A local-first RAG assistant for technical engineering documents. The current
application loads PDFs, splits them into chunks, embeds them with a Sentence
Transformers model, stores them in FAISS, and uses an Ollama-hosted LLM through
LangChain tools.

## Current baseline

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, normalized
- Chunking: 512 characters with 64-character overlap
- Retrieval: MMR, `k=4`, `fetch_k=max(10, 2*k)`
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
