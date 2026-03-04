.PHONY: install setup-models lock test ingest build-corpus build-corpus-llm-batch build-corpus-simple eval-strategies mock-oracle debug-graph llm-triples-poc collect-and-build-graph build-graph-from-corpus

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
ingest:
	poetry run python scripts/01_ingest_data.py \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Relation extraction: RELATION_EXTRACTOR=rebel (default) or llm-batch (two-stage Discovery+Validation, waits for batches)
build-corpus:
	poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Build corpus with two-stage LLM extraction (Discovery -> Validation). Blocks until complete.
build-corpus-llm-batch:
	RELATION_EXTRACTOR=llm-batch $(MAKE) build-corpus

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

# Dev: run two-stage LLM extraction on one input. Pass TEXT=... or --text-file path (e.g. make llm-triples-poc TEXT="Eva Busch was a cabaret artist.")
llm-triples-poc:
	poetry run python scripts/dev/run_llm_triple_extractor.py "$(TEXT)" --output-dir "$(OUTPUT_DIR)"

# Run two-stage LLM extraction + build graph. Resume-safe; waits for batches by default. Use when corpus.jsonl exists.
collect-and-build-graph:
	poetry run python scripts/corpus/run_llm_triple_batch_2stage.py \
		--corpus "$(OUTPUT_DIR)/corpus.jsonl" \
		--output-dir "$(OUTPUT_DIR)"

# Rebuild graph from a corpus file. CORPUS=path GRAPH=path (e.g. corpus.jsonl -> graph.pkl)
build-graph-from-corpus:
	poetry run python scripts/corpus/build_graph_from_corpus.py \
		--corpus "$(CORPUS)" \
		--graph-out "$(GRAPH)"