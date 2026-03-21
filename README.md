# RAQR (Reasoning-Aware Query Router)

RAQR is a query router designed to route QA queries to the most appropriate retrieval strategy (e.g., Dense Retrieval vs. Graph-based Reasoning).

## Setup & Dependencies

1. **Install dependencies:**
   Make sure to have Python 3.10+ installed. Then run:
   ```bash
   make install
   ```

2. **Download required local models (run once):**
   ```bash
   make setup-models
   ```
   This downloads the local `SentenceTransformer` (`all-MiniLM-L6-v2`), spaCy language models, and tiktoken encodings.

3. **Configure the Environment:**
   Create a `.env` file in the root directory (or inject via your environment). Most importantly, set your OpenAI API key for information extraction and strategy generation:
   ```env
   OPENAI_API_KEY=your_sk_key
   # Example data sources (used by make ingest)
   NQ_PATH=data/raw/nq_samples.jsonl
   2WIKI_PATH=data/raw/2wiki_samples.jsonl
   ```

## The Incremental Pipeline

RAQR has an end-to-end data pipeline that transforms raw datasets into a fully labeled "router dataset" capable of training classification models. 

**Important:** The pipeline is designed to be fully **incremental**. It "fills in the gaps." If you process 1,000 questions and later decide to add another 2,000, running the pipeline commands again will *not* overwrite your existing data or re-evaluate already-answered questions. It will only process the novel data and merge the new results, saving significant time and OpenAI API costs.

Here are the four core steps:

### 1. `make ingest`
Combines your raw datasets (NQ, 2Wiki) into a unified `data/processed/benchmark.jsonl`.
- **Incremental:** It looks for an existing `benchmark.jsonl` and dedupes incoming data by `question_id`. Only novel questions are appended and assigned to Train/Dev/Test splits. Old splits are preserved.

### 2. `make build-corpus`
Downloads required Wikipedia documents, chunks the text, and runs an OpenAI Batch API pass to extract entities and relations into `data/processed/corpus.jsonl`. Afterward, it builds local FAISS and Graph indexes.
- **Incremental:** It loads existing corpus chunks into a cache. If a chunk already has extracted `entities` and `relations`, it skips sending that chunk back to the OpenAI Batch API. Fresh chunks get batched, and all chunks are safely merged.

### 3. `make run-strategy-batch`
For every question in your benchmark, RAQR generates an answer using a Dense Strategy and a Graph Strategy via the OpenAI Batch API, saving the raw scores/answers to `data/processed/oracle_raw_scores.jsonl`.
- **Incremental:** The script reads any existing `oracle_raw_scores.jsonl`. If a question has already been answered by both strategies, it is excluded from the new batch.

### 4. `make build-router-dataset`
Scores the strategy answers (EM/F1), determines the "winning" strategy ("oracle label"), and calculates local model features (Sentence Embeddings, FAISS probe features, linguistic Q-features) to build `data/training/labeled_{split}.jsonl`.
- **Incremental:** It reads your existing `labeled_*.jsonl` files. For questions that already have their embeddings and probe signals computed, it skips the heavy local compute and only calculates features for newly-added questions.

---

### Running it End-to-End

If your `.env` is set up with new raw data, just repeatedly run:
```bash
make ingest
make build-corpus
make run-strategy-batch
make build-router-dataset
```
*(Note for batch commands: Ensure you wait for the OpenAI batches to complete. The `run_*` commands orchestrate both submit and collect.)*
