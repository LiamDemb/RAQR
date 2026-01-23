# RAQR (Reasoning-Aware Query Router)

RAQR is a query router that can be used to route QA queries to the most appropriate retrieval stategy.

## Setup Notes

- Install dependencies with Poetry.
- Download the spaCy English model (required for entity extraction):
  - `python -m spacy download en_core_web_sm`

## Convenience (Makefile + .env)

- A `Makefile` is included for common tasks: `make install`, `make test`,
  `make ingest`, `make build-corpus`.
- You can optionally create a `.env` file to set dataset paths and defaults:
  - `NQ_PATH`, `COMPLEXTEMPQA_PATH`, `WIKIWHY_PATH`, `BENCHMARK_PATH`
  - `OUTPUT_DIR`, `DOCSTORE_PATH`, `MODEL_NAME`
  - `MAX_PAGES`, `MAX_HOPS`, `MAX_LIST_PAGES`, `MAX_COUNTRY_PAGES`
  - `SEED`, `TRAIN_RATIO`, `DEV_RATIO`, `TEST_RATIO`
  - `NQ_VERSION`, `COMPLEXTEMPQA_VERSION`, `WIKIWHY_VERSION`