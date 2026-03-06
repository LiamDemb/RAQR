# RAQR (Reasoning-Aware Query Router)

RAQR is a query router that can be used to route QA queries to the most appropriate retrieval strategy.

## First-time setup

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Download required models** (requires network access once):
   ```bash
   make setup-models
   ```
   This downloads: SentenceTransformer `all-MiniLM-L6-v2`, spaCy, and tiktoken. For the LLM-driven pipeline, set `OPENAI_API_KEY` and ensure `flashtext` is installed.

3. **Environment:** Set `OPENAI_API_KEY` for corpus building (LLM information extraction) and query-time entity extraction.

## Convenience (Makefile + .env)

- A `Makefile` is included for common tasks: `make install`, `make test`,
  `make ingest`, `make build-corpus` (ingestion + chunking + LLM IE batch).
- You can optionally create a `.env` file to set dataset paths and defaults:
  - `NQ_PATH`, `COMPLEXTEMPQA_PATH`, `WIKIWHY_PATH`, `HOTPOTQA_PATH`, `BENCHMARK_PATH`
  - `OUTPUT_DIR`, `DOCSTORE_PATH`, `MODEL_NAME`
  - `SPACY_MODEL` (default: `en_core_web_sm`)
  - `LLM_ONEPASS_MODEL`, `LLM_ONEPASS_MAX_TOKENS` (IE extraction)
  - `MAX_PAGES`, `MAX_HOPS`, `MAX_LIST_PAGES`, `MAX_COUNTRY_PAGES`
  - `CHUNK_MIN_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`
  - `SEED`, `TRAIN_RATIO`, `DEV_RATIO`, `TEST_RATIO`
  - `NQ_VERSION`, `COMPLEXTEMPQA_VERSION`, `WIKIWHY_VERSION`