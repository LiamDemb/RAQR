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
   This downloads: spaCy `en_core_web_sm`, SentenceTransformer `all-MiniLM-L6-v2`, and REBEL `Babelscape/rebel-large`. Models are cached and can be reused offline.

3. **Optional:** Set `HF_HOME` and `TRANSFORMERS_CACHE` to control HuggingFace cache location. Default: `data/processed/hf_cache`.

## Convenience (Makefile + .env)

- A `Makefile` is included for common tasks: `make install`, `make test`,
  `make ingest`, `make build-corpus`.
- You can optionally create a `.env` file to set dataset paths and defaults:
  - `NQ_PATH`, `COMPLEXTEMPQA_PATH`, `WIKIWHY_PATH`, `BENCHMARK_PATH`
  - `OUTPUT_DIR`, `DOCSTORE_PATH`, `MODEL_NAME`, `RE_MODEL_NAME`
  - `SPACY_MODEL` (default: `en_core_web_sm`)
  - `MAX_PAGES`, `MAX_HOPS`, `MAX_LIST_PAGES`, `MAX_COUNTRY_PAGES`
  - `SEED`, `TRAIN_RATIO`, `DEV_RATIO`, `TEST_RATIO`
  - `NQ_VERSION`, `COMPLEXTEMPQA_VERSION`, `WIKIWHY_VERSION`