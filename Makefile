.PHONY: install lock test ingest build-corpus build-corpus-simple

-include .env

NQ_PATH ?= data/raw/nq_300.jsonl
COMPLEXTEMPQA_PATH ?= data/raw/complex_tempqa_300.jsonl
WIKIWHY_PATH ?= data/raw/wikiwhy_300.jsonl
BENCHMARK_PATH ?= data/processed/benchmark.jsonl
OUTPUT_DIR ?= data/processed
HF_HOME ?= $(OUTPUT_DIR)/hf_cache

install:
	poetry install

lock:
	poetry lock

test:
	poetry run pytest

ingest:
	poetry run python scripts/01_ingest_data.py \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

build-corpus:
	poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

build-corpus-simple:
	HF_HOME="$(HF_HOME)" poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--max-pages 3 \
		--max-hops 1 \
		--max-list-pages 0 \
		--max-country-pages 0 \
		--re-batch-size 1 \
		--re-max-input-chars 600 \
		--re-max-new-tokens 64
