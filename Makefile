.PHONY: install setup-models lock test ingest build-corpus build-corpus-simple eval-strategies mock-oracle debug-graph

-include .env

NQ_PATH ?= data/raw/nq_300.jsonl
COMPLEXTEMPQA_PATH ?= data/raw/complex_tempqa_300.jsonl
WIKIWHY_PATH ?= data/raw/wikiwhy_300.jsonl
HOTPOTQA_PATH ?= data/raw/hotpotqa_300.jsonl
BENCHMARK_PATH ?= data/processed/benchmark.jsonl
OUTPUT_DIR ?= data/processed
HF_HOME ?= $(OUTPUT_DIR)/hf_cache

install:
	poetry install

setup-models:
	HF_HOME="$(HF_HOME)" poetry run python scripts/00_setup_models.py

lock:
	poetry lock

test:
	poetry run pytest

# Dataset paths: set in .env or override on CLI. Omit a path (e.g. NQ_PATH=) to exclude that dataset.
# Example single-dataset ingest: NQ_PATH= COMPLEXTEMPQA_PATH= WIKIWHY_PATH= make ingest
ingest:
	poetry run python scripts/01_ingest_data.py \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

build-corpus:
	poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

build-corpus-simple:
	HF_HOME="$(HF_HOME)" poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--max-pages 3 \
		--max-hops 1 \
		--max-list-pages 0 \
		--max-country-pages 0 \
		--re-batch-size 1 \
		--re-max-input-tokens 512 \
		--re-max-new-tokens 64

eval-strategies:
	poetry run python scripts/dev/evaluate_strategies.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--use-llm-judge

mock-oracle:
	poetry run python scripts/dev/mock_oracle_eval.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Debug GraphStrategy: pass your query via QUERY=... (e.g. make debug-graph QUERY="What happened in 2017?")
debug-graph:
	poetry run python scripts/strategies/debug_graph.py "$(QUERY)" --output-dir "$(OUTPUT_DIR)"
